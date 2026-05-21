#!/usr/bin/env python3
"""
Minimal config-driven fine-tuning for Waypoint models.

Usage:
    python finetune.py --config configs/finetune_classification.yaml \\
        --model outpost-bio/Waypoint-6m \\
        --data path/to/prepared_dataset.parquet \\
        --output_dir outputs/finetune_classification \\
        --task_type classification \\
        --target DiseaseStatus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, EarlyStoppingCallback, Trainer, TrainingArguments

from src.dataset import (
    MicrobiomeBenchmarkDataset,
    build_drug_map,
    build_label_maps,
    load_waypoint_dataframe,
    try_load_token_std_means,
)
from src.models import ClassificationModel, RegressionModel
from src.scoring import predictions_to_arrays, score_task
from src.tokenizer import load_tokenizer

DEFAULTS = {
    "split_column": None,
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "max_length": 512,
    "pooling_strategy": "last_token",
    "filter_unk_taxa": True,
    "seed": 42,
    "learning_rate": 3e-5,
    "num_epochs": 5,
    "batch_size": 32,
    "warmup_steps": 100,
    "weight_decay": 0.001,
    "eval_strategy": "steps",
    "eval_steps": 400,
    "logging_steps": 5,
    "patience": 5,
    "save_total_limit": 1,
    "use_lora": False,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["c_attn", "c_proj"],
    "lora_bias": "none",
    "lora_fan_in_fan_out": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Waypoint model")
    parser.add_argument(
        "--model", required=True, help="HF model id or local model path"
    )
    parser.add_argument(
        "--data", required=True, help="Prepared waypoint-format data file"
    )
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["classification", "regression"],
        help="Fine-tuning task type",
    )
    parser.add_argument("--target", required=True, help="Target column in --data")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to flat fine-tuning YAML config",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **cfg}


def maybe_apply_lora(base_model, cfg: dict[str, Any]):
    if not cfg["use_lora"]:
        return base_model

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias=cfg["lora_bias"],
        fan_in_fan_out=cfg["lora_fan_in_fan_out"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {list(df.columns)}"
        )


def prepare_dataframe(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    targets = cfg["targets"]
    require_columns(df, ["Taxa", "Relative Abundances", *targets])

    df = df.copy()
    if cfg["task_type"] == "classification":
        df = df.dropna(subset=targets)
        for target in targets:
            df[target] = df[target].astype(str)
    else:
        for target in targets:
            df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=targets)
    return df.reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    split_column = cfg["split_column"]
    if split_column:
        require_columns(df, [split_column])
        split = df[split_column].astype(str).str.strip().str.lower()
        train_df = df[split.isin(["train", "training"])].copy()
        val_df = df[split.isin(["validation", "valid", "val", "dev"])].copy()
        test_df = df[split == "test"].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("split_column must include train and validation rows")
        return train_df, val_df, test_df if len(test_df) else None

    val_fraction = float(cfg["val_fraction"])
    test_fraction = float(cfg["test_fraction"])
    if val_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Use val_fraction > 0 and val_fraction + test_fraction < 1")

    seed = cfg["seed"]
    if test_fraction > 0:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_val_df, test_df = df, None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction / (1 - test_fraction),
        random_state=seed,
        shuffle=True,
    )
    return train_df, val_df, test_df


def validate_targets(train_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    for target in cfg["targets"]:
        if train_df[target].nunique() < 2:
            raise ValueError(f"Training split needs at least two values for {target!r}")


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        vals = torch.stack([sample[key] for sample in batch])
        out[key] = vals.float() if key in ("targets", "drug_onehot") else vals
    return out


def get_class_weights(
    train_df: pd.DataFrame,
    targets: list[str],
    label_maps: dict[str, dict[str, int]],
) -> list[torch.Tensor]:
    weights_list = []
    for target in targets:
        labels = np.asarray(train_df[target].map(label_maps[target]).astype(int))
        n_classes = len(label_maps[target])
        classes_present = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes_present, y=labels)

        weight_tensor = torch.ones(n_classes, dtype=torch.float32)
        for cls, weight in zip(classes_present, weights):
            weight_tensor[cls] = weight
        weights_list.append(weight_tensor)
    return weights_list


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def evaluate(
    trainer: Trainer,
    dataset: MicrobiomeBenchmarkDataset,
    cfg: dict[str, Any],
    output_path: Path,
) -> tuple[float, dict]:
    out = trainer.predict(dataset)
    y_true, y_pred, y_prob = predictions_to_arrays(
        out.predictions,
        out.label_ids,
        cfg["task_type"],
        len(cfg["targets"]),
    )
    score, metrics = score_task(
        y_true,
        y_pred,
        cfg["targets"],
        cfg["task_type"],
        y_prob_list=y_prob if cfg["task_type"] == "classification" else None,
    )
    save_json(output_path, {"score": score, "metrics": metrics})
    return score, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.update(
        {
            "model": args.model,
            "data": args.data,
            "output_dir": args.output_dir,
            "task_type": args.task_type,
            "targets": [args.target],
        }
    )
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    print(f"Loading data from {cfg['data']} ...")
    df = prepare_dataframe(load_waypoint_dataframe(cfg["data"]), cfg)
    train_df, val_df, test_df = split_dataframe(df, cfg)
    validate_targets(train_df, cfg)
    print(
        f"Split sizes: {len(train_df)} train / {len(val_df)} validation"
        + (f" / {len(test_df)} test" if test_df is not None else "")
    )

    print(f"Loading model from {cfg['model']} ...")
    tokenizer = load_tokenizer(cfg["model"])
    token_std_means = try_load_token_std_means(cfg["model"])
    base_model = AutoModel.from_pretrained(cfg["model"], trust_remote_code=True)
    base_model = maybe_apply_lora(base_model, cfg)

    label_maps = None
    class_weights = None
    drug_map = build_drug_map(train_df) if "Drug" in train_df.columns else None
    if cfg["task_type"] == "classification":
        label_maps = build_label_maps(train_df, cfg["targets"])
        class_weights = get_class_weights(train_df, cfg["targets"], label_maps)
        print(f"Label maps: {label_maps}")
    if drug_map is not None:
        print(f"Drug categories: {len(drug_map)}")

    dataset_kwargs = dict(
        tokenizer=tokenizer,
        targets=cfg["targets"],
        task_type=cfg["task_type"],
        label_maps=label_maps,
        drug_map=drug_map,
        max_length=cfg["max_length"],
        token_std_means=token_std_means,
        filter_unk_taxa=cfg["filter_unk_taxa"],
    )
    train_ds = MicrobiomeBenchmarkDataset(train_df, **dataset_kwargs)
    val_ds = MicrobiomeBenchmarkDataset(val_df, **dataset_kwargs)
    test_ds = (
        MicrobiomeBenchmarkDataset(test_df, **dataset_kwargs)
        if test_df is not None
        else None
    )

    if cfg["task_type"] == "classification":
        model = ClassificationModel(
            base_model=base_model,
            tokenizer=tokenizer,
            label_dims=train_ds.label_dims,
            pooling_strategy=cfg["pooling_strategy"],
            drug_dim=len(drug_map) if drug_map else 0,
            class_weights=class_weights,
        )
        label_names = ["labels"]
    else:
        model = RegressionModel(
            base_model=base_model,
            tokenizer=tokenizer,
            num_targets=len(cfg["targets"]),
            pooling_strategy=cfg["pooling_strategy"],
            drug_dim=len(drug_map) if drug_map else 0,
        )
        label_names = ["targets"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        label_names=label_names,
        disable_tqdm=False,
    )
    callbacks = []
    if cfg["patience"] > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["patience"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    print("Fine-tuning ...")
    trainer.train()

    print("Evaluating ...")
    val_score, val_metrics = evaluate(
        trainer,
        val_ds,
        cfg,
        output_dir / "validation_metrics.json",
    )
    test_score, test_metrics = None, None
    if test_ds is not None:
        test_score, test_metrics = evaluate(
            trainer,
            test_ds,
            cfg,
            output_dir / "test_metrics.json",
        )

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)
    model.base_model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    if token_std_means is not None:
        token_std_means.to_parquet(best_model_dir / "token_std_means.parquet")
    torch.save(model.state_dict(), best_model_dir / "finetuned_model_state.pt")

    results = {
        "config": cfg,
        "label_maps": label_maps,
        "drug_map": drug_map,
        "validation_score": val_score,
        "validation_metrics": val_metrics,
        "test_score": test_score,
        "test_metrics": test_metrics,
    }
    save_json(output_dir / "finetune_results.json", results)
    save_json(best_model_dir / "finetune_config.json", results)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
