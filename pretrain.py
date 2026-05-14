#!/usr/bin/env python3
"""
Pretrain a GPT2 causal language model on the public microbiome pretraining dataset.

Downloads data from HuggingFace Hub (outpost-bio/Atlas), builds a
TaxonomicTokenizer, and trains with next-token prediction.

Usage:
    python pretrain.py \\
        --model_config configs/models/gpt2-6m-mgm.yaml \\
        --pretrain_config configs/pretraining/gpt2.yaml \\
        --output_dir outputs/pretrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from src.dataset import (
    MicrobiomePretrainingDataset,
    compute_token_std_means,
    load_waypoint_dataframe,
)
from src.tokenizer import TaxonomicTokenizer

HF_DATASET = "outpost-bio/Atlas"


def main():
    parser = argparse.ArgumentParser(description="Pretrain a microbiome language model")
    parser.add_argument(
        "--model_config",
        default="configs/models/gpt2-6m-mgm.yaml",
        help="Path to model architecture config YAML (in configs/models/)",
    )
    parser.add_argument(
        "--pretrain_config",
        default="configs/pretraining.yaml",
        help="Path to pretraining hyperparameter config YAML (in configs/pretraining/)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/pretrain",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of training samples (for quick testing)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Path to a local waypoint-format file (.parquet/.csv/.tsv) with "
            "'Taxa' and 'Relative Abundances' columns. If omitted, downloads "
            f"{HF_DATASET} from the HuggingFace Hub."
        ),
    )
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.pretrain_config) as f:
        train_cfg = yaml.safe_load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pretraining data (local file or HuggingFace Hub)
    # ------------------------------------------------------------------
    if args.data is not None:
        print(f"Loading dataset from {args.data} ...")
        df = load_waypoint_dataframe(args.data)
    else:
        print(f"Loading dataset from {HF_DATASET} ...")
        ds = load_dataset(HF_DATASET, split="pretrain")
        df = ds.to_pandas()

    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Using {len(df)} samples (--max_samples)")

    print(f"Loaded {len(df):,} samples")

    # ------------------------------------------------------------------
    # 2. Build tokenizer from the data
    # ------------------------------------------------------------------
    print("Building tokenizer ...")
    unique_taxa: set[str] = set()
    for taxa_list in tqdm(
        df["Taxa"],
        total=len(df),
        desc="Collecting unique taxa",
        unit="sample",
    ):
        unique_taxa.update(str(t) for t in taxa_list)

    tokenizer = TaxonomicTokenizer(
        taxa=sorted(unique_taxa),
        rank=train_cfg.get("taxon_rank"),
        fallback_to_higher_rank=train_cfg.get("fallback_to_higher_rank", True),
    )
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 3. Compute token statistics for z-score ordering
    # ------------------------------------------------------------------
    print("Computing token statistics ...")
    token_std_means = compute_token_std_means(
        df, tokenizer, show_progress=True, progress_desc="Token statistics"
    )
    token_std_means.to_parquet(best_model_dir / "token_std_means.parquet")
    print(f"Computed stats for {len(token_std_means)} tokens")

    # ------------------------------------------------------------------
    # 4. Create tokenised datasets (train / val split)
    # ------------------------------------------------------------------
    val_split = train_cfg.get("val_split", 0.1)
    n_val = int(len(df) * val_split)
    df_shuffled = df.sample(frac=1, random_state=train_cfg.get("seed", 42)).reset_index(
        drop=True
    )
    df_train = df_shuffled.iloc[n_val:]
    df_val = df_shuffled.iloc[:n_val]

    max_length = train_cfg.get("max_length", 512)
    print(f"Tokenising {len(df_train):,} train / {len(df_val):,} val samples ...")
    train_ds = MicrobiomePretrainingDataset(
        df_train,
        tokenizer,
        max_length=max_length,
        token_std_means=token_std_means,
        show_progress=True,
        progress_desc="Tokenising train",
    )
    val_ds = MicrobiomePretrainingDataset(
        df_val,
        tokenizer,
        max_length=max_length,
        token_std_means=token_std_means,
        show_progress=True,
        progress_desc="Tokenising val",
    )
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ------------------------------------------------------------------
    # 5. Initialise model
    # ------------------------------------------------------------------
    gpt2_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=model_cfg["n_positions"],
        n_embd=model_cfg["n_embd"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(gpt2_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT2 with {n_params / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        logging_dir=str(output_dir / "logs"),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        warmup_steps=train_cfg["warmup_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=True,
        report_to=[],
        disable_tqdm=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_cfg["patience"])
        ],
    )

    print("Starting pretraining ...")
    trainer.train()
    trainer.save_model(str(best_model_dir))
    print(f"Best model saved to {best_model_dir}")

    # Save configs for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": model_cfg, "training": train_cfg}, f, indent=2)

    print("Pretraining complete!")


if __name__ == "__main__":
    main()
