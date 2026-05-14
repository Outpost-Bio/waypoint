#!/usr/bin/env python3
"""
Generate sample embeddings using a pretrained Waypoint model.

Input is a waypoint-format file (parquet/csv/tsv) with 'Taxa' and
'Relative Abundances' columns. To start from a raw abundance matrix, run
``prepare_dataset.py`` first to serialize it into this format.

Usage:
    python embed.py \\
        --model outpost-bio/Waypoint-6m \\
        --data samples.parquet \\
        --output embeddings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.dataset import (
    _sort_by_abundance,
    _sort_by_zscore,
    load_waypoint_dataframe,
    try_load_token_std_means,
)
from src.models import _pool
from src.tokenizer import load_tokenizer


def tokenize_for_embedding(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    token_std_means: pd.DataFrame | None,
) -> list[dict[str, torch.Tensor]]:
    """Tokenise each row into [BOS] + sorted_token_ids + [EOS], padded.

    Mirrors ``MicrobiomePretrainingDataset`` but omits ``labels`` and
    preserves one output row per input row (so the caller can keep
    sample IDs aligned even when a row has no known taxa).
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    unk_id = tokenizer.unk_token_id

    samples: list[dict[str, torch.Tensor]] = []
    for _, row in df.iterrows():
        taxa = row["Taxa"]
        ras = row["Relative Abundances"]

        token_ids: list[int] = []
        ra_vals: list[float] = []
        if hasattr(taxa, "__iter__") and not isinstance(taxa, str):
            for taxon, ra in zip(taxa, ras):
                tid = tokenizer._convert_token_to_id(str(taxon))
                if tid == unk_id:
                    continue
                token_ids.append(tid)
                ra_vals.append(float(ra))

        if token_ids:
            if token_std_means is not None:
                token_ids = _sort_by_zscore(
                    token_ids, ra_vals, token_std_means, tokenizer
                )
            else:
                token_ids = _sort_by_abundance(token_ids, ra_vals)

        seq = [bos_id] + token_ids[: max_length - 2] + [eos_id]
        attention_mask = [1] * len(seq)
        pad_len = max_length - len(seq)
        seq = seq + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        samples.append(
            {
                "input_ids": torch.tensor(seq, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )
    return samples


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Generate Waypoint embeddings")
    parser.add_argument(
        "--model",
        default="outpost-bio/Waypoint-6m",
        help="HuggingFace model id or local path to a pretrained Waypoint model",
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Path to a waypoint-format file (.parquet/.csv/.tsv) with 'Taxa' "
            "and 'Relative Abundances' columns. Run prepare_dataset.py first "
            "if you have an abundance matrix."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for embeddings (.parquet or .csv)",
    )
    parser.add_argument(
        "--pooling",
        default="last_token",
        choices=["mean", "first_token", "last_token", "cls_token"],
        help="How to pool token hidden states into a single embedding per sample",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps; auto-detected if omitted",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    print(f"Loading waypoint-format data from {data_path} ...")
    df = load_waypoint_dataframe(data_path)
    print(f"Loaded {len(df):,} samples")

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer + token statistics
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model} ...")
    tokenizer = load_tokenizer(args.model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    device = _resolve_device(args.device)
    model.to(device)
    print(f"Using device: {device}")

    token_std_means = try_load_token_std_means(args.model)
    if token_std_means is not None:
        print(f"Loaded token_std_means ({len(token_std_means)} tokens)")
    else:
        print("No token_std_means found; using descending abundance order")

    # ------------------------------------------------------------------
    # 3. Tokenise
    # ------------------------------------------------------------------
    print("Tokenising ...")
    samples = tokenize_for_embedding(df, tokenizer, args.max_length, token_std_means)

    # ------------------------------------------------------------------
    # 4. Forward pass + pool
    # ------------------------------------------------------------------
    print("Generating embeddings ...")
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False)
    all_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            pooled = _pool(hidden, attention_mask, args.pooling)
            all_embeddings.append(pooled.cpu())
    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(
        embeddings,
        index=df.index,
        columns=[f"dim_{i}" for i in range(embeddings.shape[1])],
    )
    out_df.index.name = df.index.name or "sample_id"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        out_df.to_csv(output_path)
    else:
        out_df.to_parquet(output_path)
    print(f"Saved embeddings of shape {embeddings.shape} to {output_path}")


if __name__ == "__main__":
    main()
