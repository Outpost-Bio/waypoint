#!/usr/bin/env python3
"""
Benchmark a pretrained microbiome model on the 8 Compass tasks.

Loads the model from HuggingFace Hub (or a local path), downloads benchmark
data from outpost-bio/Compass, fine-tunes on each task, and reports the
final benchmark score.

Usage:
    # Benchmark the published model
    python benchmark.py --model outpost-bio/Waypoint-6m-mgm --output_dir outputs/benchmark

    # Benchmark a locally pretrained model
    python benchmark.py --model outputs/pretrain/best_model --output_dir outputs/benchmark

    # Run a single task for quick testing
    python benchmark.py --model outpost-bio/Waypoint-6m-mgm --tasks 1 --output_dir outputs/benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.dataset import (
    MicrobiomeBenchmarkDataset,
    build_drug_map,
    build_label_maps,
    try_load_token_std_means,
)
from src.models import ClassificationModel, RegressionModel
from src.scoring import predictions_to_arrays, score_task
from src.tokenizer import load_tokenizer

# ---------------------------------------------------------------------------
# HuggingFace Hub identifiers
# ---------------------------------------------------------------------------

HF_BENCHMARK_DATASET = "outpost-bio/Compass"

# ---------------------------------------------------------------------------
# Task definitions: (name, hub_config, pre_filter, features, targets, task_type)
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name": "1_biome",
        "hub_config": "mgnify-biomes",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["Biome 1", "Biome 2", "Biome 3", "Biome 4", "Biome 5"],
        "task_type": "classification",
        "pre_filter": None,
    },
    {
        "name": "2_biome_gut",
        "hub_config": "mgnify-biomes",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["Biome 4", "Biome 5"],
        "task_type": "classification",
        "pre_filter": lambda df: df[df["Biome 3"] == "Digestive system"],
    },
    {
        "name": "3_sic",
        "hub_config": "handuo",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["SIC Name"],
        "task_type": "classification",
        "pre_filter": lambda df: df[
            df["SIC Name"].str.startswith("SIC", na=False)
            & ~df["SIC Name"].str.contains("control", case=False, na=False)
            & ~df["SIC Name"].str.contains("seed", case=False, na=False)
        ],
    },
    {
        "name": "4_drug_non_drug",
        "hub_config": "handuo",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["Control"],
        "task_type": "classification",
        "pre_filter": None,
    },
    {
        "name": "5_drug_class",
        "hub_config": "handuo",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["ATC Class"],
        "task_type": "classification",
        "pre_filter": lambda df: df[df["ATC Class"].notna()],
    },
    {
        "name": "6_drug_degradation",
        "hub_config": "mastrorilli",
        "features": ["Taxa", "Relative Abundances", "Drug"],
        "targets": ["Degradation Rate"],
        "task_type": "regression",
        "pre_filter": None,
    },
    {
        "name": "7_infant_age",
        "hub_config": "roswall",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["Timepoint"],
        "task_type": "classification",
        "pre_filter": None,
    },
    {
        "name": "8_birth_mode",
        "hub_config": "roswall",
        "features": ["Taxa", "Relative Abundances"],
        "targets": ["Delivery Mode"],
        "task_type": "classification",
        "pre_filter": None,
    },
]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = torch.stack([b[k] for b in batch])
        if k in ("targets", "drug_onehot"):
            vals = vals.float()
        out[k] = vals
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_class_weights(
    train_df: pd.DataFrame,
    targets: list[str],
    label_maps: dict[str, dict[str, int]],
) -> list[torch.Tensor]:
    weights_list = []
    for t in targets:
        labels = np.asarray(train_df[t].map(label_maps[t]).dropna().astype(int))
        n_classes = len(label_maps[t])
        if len(labels) == 0:
            weights_list.append(torch.ones(n_classes, dtype=torch.float32))
            continue
        classes_present = np.unique(labels)
        w = compute_class_weight("balanced", classes=classes_present, y=labels)
        weight_tensor = torch.ones(n_classes, dtype=torch.float32)
        for cls, wt in zip(classes_present, w):
            weight_tensor[cls] = wt
        weights_list.append(weight_tensor)
    return weights_list


def load_task_data(
    task_def: dict,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from HuggingFace Hub for a task."""
    ds = load_dataset(HF_BENCHMARK_DATASET, task_def["hub_config"])

    dfs = {}
    for split_name, hf_split in [
        ("train", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ]:
        split_df = ds[hf_split].to_pandas()
        # Apply pre_filter before column selection so filters can use columns that are
        # not in features/targets (e.g. 2_biome_gut filters on Biome 3).
        if task_def["pre_filter"] is not None:
            split_df = task_def["pre_filter"](split_df)

        cols = task_def["features"] + task_def["targets"]
        available_cols = [c for c in cols if c in split_df.columns]
        split_df = split_df[available_cols].copy()

        if max_samples is not None:
            split_df = split_df.head(max_samples)

        dfs[split_name] = split_df.reset_index(drop=True)

    return dfs["train"], dfs["validation"], dfs["test"]


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------


def run_task(
    task_def: dict,
    model_path: str,
    tokenizer,
    token_std_means: pd.DataFrame | None,
    benchmark_cfg: dict,
    output_dir: Path,
    max_samples: int | None = None,
) -> dict:
    """Fine-tune and evaluate on a single benchmark task."""
    task_name = task_def["name"]
    task_type = task_def["task_type"]
    targets = task_def["targets"]
    has_drug = "Drug" in task_def["features"]

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name} ({task_type})")
    print(f"{'=' * 60}")

    # Load data
    train_df, val_df, test_df = load_task_data(task_def, max_samples=max_samples)
    print(f"  Data: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(train_df.columns)

    # Build maps from training data
    label_maps = None
    drug_map = None
    if task_type == "classification":
        label_maps = build_label_maps(train_df, targets)
    if has_drug:
        drug_map = build_drug_map(train_df)

    # Create datasets
    ds_kwargs = dict(
        tokenizer=tokenizer,
        targets=targets,
        task_type=task_type,
        label_maps=label_maps,
        drug_map=drug_map,
        max_length=512,
        token_std_means=token_std_means,
        filter_unk_taxa=benchmark_cfg.get("filter_unk_taxa", True),
    )
    train_ds = MicrobiomeBenchmarkDataset(train_df, **ds_kwargs)
    val_ds = MicrobiomeBenchmarkDataset(val_df, **ds_kwargs)
    test_ds = MicrobiomeBenchmarkDataset(test_df, **ds_kwargs)
    print(
        f"  Tokenised: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test"
    )

    # Build model
    base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    if task_type == "regression":
        model = RegressionModel(
            base_model=base_model,
            tokenizer=tokenizer,
            num_targets=len(targets),
            drug_dim=len(drug_map) if drug_map else 0,
            pooling_strategy=benchmark_cfg.get("pooling_strategy", "last_token"),
        )
    else:
        assert label_maps is not None
        assert train_ds.label_dims is not None
        class_weights = get_class_weights(train_df, targets, label_maps)
        model = ClassificationModel(
            base_model=base_model,
            tokenizer=tokenizer,
            label_dims=train_ds.label_dims,
            drug_dim=len(drug_map) if drug_map else 0,
            pooling_strategy=benchmark_cfg.get("pooling_strategy", "last_token"),
            class_weights=class_weights,
        )

    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(task_output_dir),
        num_train_epochs=benchmark_cfg["num_epochs"],
        per_device_train_batch_size=benchmark_cfg["batch_size"],
        per_device_eval_batch_size=benchmark_cfg["batch_size"],
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=benchmark_cfg["eval_steps"],
        eval_steps=benchmark_cfg["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=benchmark_cfg["learning_rate"],
        warmup_steps=benchmark_cfg["warmup_steps"],
        weight_decay=benchmark_cfg["weight_decay"],
        logging_steps=benchmark_cfg["logging_steps"],
        report_to=[],
        label_names=["targets"] if task_type == "regression" else ["labels"],
        disable_tqdm=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=benchmark_cfg["patience"])
        ],
    )

    trainer.train()

    # Evaluate on test set
    test_out = trainer.predict(test_ds)
    y_true_list, y_pred_list, y_prob_list = predictions_to_arrays(
        test_out.predictions, test_out.label_ids, task_type, len(targets)
    )
    task_score, metrics = score_task(
        y_true_list,
        y_pred_list,
        targets,
        task_type,
        y_prob_list=y_prob_list if task_type == "classification" else None,
    )

    print(f"  Score: {task_score:.4f}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    return {
        "task": task_name,
        "task_type": task_type,
        "score": task_score,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark a microbiome model")
    parser.add_argument(
        "--model",
        default="outpost-bio/Waypoint-6m",
        help="HuggingFace model id or local path to pretrained model",
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/benchmark",
        help="Directory for results",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        type=int,
        default=None,
        help="Task numbers to run (1-8). Default: run all tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap each split (train/val/test) at this many samples. Useful for quick smoke tests.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        benchmark_cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Select tasks
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = [TASKS[i - 1] for i in args.tasks]

    # Load tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = load_tokenizer(args.model)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load token_std_means (for z-score ordering)
    token_std_means = try_load_token_std_means(args.model)
    if token_std_means is not None:
        print(f"Loaded token_std_means ({len(token_std_means)} tokens)")
    else:
        print("No token_std_means found; using descending abundance order")

    # Run tasks
    all_results: list[dict] = []
    for task_def in tasks_to_run:
        result = run_task(
            task_def,
            args.model,
            tokenizer,
            token_std_means,
            benchmark_cfg,
            output_dir,
            max_samples=args.max_samples,
        )
        all_results.append(result)

    # Final score
    scores = [r["score"] for r in all_results]
    final_score = float(np.mean(scores))

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    for r in all_results:
        print(f"  {r['task']:25s}  {r['score']:.4f}")
    print(f"{'=' * 60}")
    print(f"  {'Final Score':25s}  {final_score:.4f}")
    print(f"{'=' * 60}")

    # Save results
    results_file = output_dir / "benchmark_results.json"
    output = {
        "model": args.model,
        "final_score": final_score,
        "results": all_results,
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(_convert(output), f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
