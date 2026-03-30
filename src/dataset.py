"""
Microbiome dataset utilities — converts (Taxa, Relative Abundances) rows into
tokenised torch tensors for pretraining and downstream fine-tuning.

Reimplements the relevant parts of mb-core's TokenisedMicrobiomeDataset
without any private dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.tokenizer import TaxonomicTokenizer


# ---------------------------------------------------------------------------
# Token statistics (for z-score ordering)
# ---------------------------------------------------------------------------


def compute_token_std_means(
    df: pd.DataFrame,
    tokenizer: TaxonomicTokenizer,
    *,
    show_progress: bool = False,
    progress_desc: str = "Token statistics",
) -> pd.DataFrame:
    """Compute per-token mean and std of relative abundance across the dataset.

    Returns a DataFrame with columns ``taxon``, ``mean``, ``std`` — one row per
    token in the tokenizer vocabulary (excluding special tokens).
    """
    accum: dict[str, list[float]] = defaultdict(list)

    row_iter = df.iterrows()
    if show_progress:
        row_iter = tqdm(
            row_iter,
            total=len(df),
            desc=progress_desc,
            unit="sample",
        )

    for _, row in row_iter:
        taxa = row["Taxa"]
        ras = row["Relative Abundances"]
        if not hasattr(taxa, "__iter__") or isinstance(taxa, str):
            continue
        for taxon, ra in zip(taxa, ras):
            token = tokenizer._extract(str(taxon))
            if token is not None and token in tokenizer.vocab:
                accum[token].append(float(ra))

    rows = []
    for token in sorted(accum.keys()):
        vals = accum[token]
        rows.append(
            {"taxon": token, "mean": np.mean(vals), "std": max(np.std(vals), 1e-9)}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tokenised dataset
# ---------------------------------------------------------------------------


def _sort_by_zscore(
    token_ids: list[int],
    ras: list[float],
    token_std_means: pd.DataFrame,
    tokenizer: TaxonomicTokenizer,
) -> list[int]:
    """Sort token IDs by descending z-score of their relative abundance."""
    taxon_col = token_std_means.set_index("taxon")

    scored = []
    for tid, ra in zip(token_ids, ras):
        token_str = tokenizer._convert_id_to_token(tid)
        if token_str in taxon_col.index:
            mean = taxon_col.loc[token_str, "mean"]
            std = taxon_col.loc[token_str, "std"]
            zscore = (ra - mean) / std
        else:
            zscore = 0.0
        scored.append((tid, zscore))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in scored]


def _sort_by_abundance(
    token_ids: list[int],
    ras: list[float],
) -> list[int]:
    """Sort token IDs by descending relative abundance."""
    pairs = sorted(zip(token_ids, ras), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in pairs]


class MicrobiomePretrainingDataset(Dataset):
    """Torch dataset for causal-LM pretraining on microbiome samples.

    Each sample becomes: [BOS] + sorted_token_ids + [EOS], padded to max_length.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TaxonomicTokenizer,
        max_length: int = 512,
        token_std_means: pd.DataFrame | None = None,
        *,
        show_progress: bool = False,
        progress_desc: str = "Tokenising",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_std_means = token_std_means

        self.samples: list[dict[str, torch.Tensor]] = []
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        row_iter = df.iterrows()
        if show_progress:
            row_iter = tqdm(
                row_iter,
                total=len(df),
                desc=progress_desc,
                unit="sample",
            )

        for _, row in row_iter:
            taxa = row["Taxa"]
            ras = row["Relative Abundances"]
            if not hasattr(taxa, "__iter__") or isinstance(taxa, str):
                continue

            token_ids = []
            ra_vals = []
            for taxon, ra in zip(taxa, ras):
                tid = tokenizer._convert_token_to_id(str(taxon))
                if tid != tokenizer.unk_token_id:
                    token_ids.append(tid)
                    ra_vals.append(float(ra))

            if not token_ids:
                continue

            # Sort tokens
            if token_std_means is not None:
                token_ids = _sort_by_zscore(
                    token_ids, ra_vals, token_std_means, tokenizer
                )
            else:
                token_ids = _sort_by_abundance(token_ids, ra_vals)

            # Build sequence: [BOS] + tokens + [EOS]
            seq = [bos_id] + token_ids[: max_length - 2] + [eos_id]
            attention_mask = [1] * len(seq)

            # Pad
            pad_len = max_length - len(seq)
            seq = seq + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

            self.samples.append(
                {
                    "input_ids": torch.tensor(seq, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(seq, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class MicrobiomeBenchmarkDataset(Dataset):
    """Torch dataset for fine-tuning on benchmark tasks (classification / regression).

    Each sample becomes: [BOS] + sorted_token_ids + [EOS], padded to max_length,
    plus labels/targets and optional drug one-hot features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TaxonomicTokenizer,
        targets: list[str],
        task_type: Literal["classification", "regression"],
        label_maps: dict[str, dict[str, int]] | None = None,
        drug_map: dict[str, int] | None = None,
        max_length: int = 512,
        token_std_means: pd.DataFrame | None = None,
        filter_unk_taxa: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.targets = targets
        self.label_maps = label_maps
        self.drug_dim = len(drug_map) if drug_map else 0

        # For ClassificationModel: number of classes per target
        self.label_dims: list[int] | None = None
        if task_type == "classification" and label_maps is not None:
            self.label_dims = [len(label_maps[t]) for t in targets]
        self.n_labels = len(targets)

        self.samples: list[dict[str, torch.Tensor]] = []
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        unk_id = tokenizer.unk_token_id

        for _, row in df.iterrows():
            taxa = row["Taxa"]
            ras = row["Relative Abundances"]
            if not hasattr(taxa, "__iter__") or isinstance(taxa, str):
                continue

            token_ids = []
            ra_vals = []
            for taxon, ra in zip(taxa, ras):
                tid = tokenizer._convert_token_to_id(str(taxon))
                if filter_unk_taxa and tid == unk_id:
                    continue
                token_ids.append(tid)
                ra_vals.append(float(ra))

            if not token_ids:
                # Pad-only sample
                token_ids = []
                ra_vals = []

            # Sort tokens
            if token_ids:
                if token_std_means is not None:
                    token_ids = _sort_by_zscore(
                        token_ids, ra_vals, token_std_means, tokenizer
                    )
                else:
                    token_ids = _sort_by_abundance(token_ids, ra_vals)

            # Build sequence: [BOS] + tokens + [EOS]
            seq = [bos_id] + token_ids[: max_length - 2] + [eos_id]
            attention_mask = [1] * len(seq)

            # Pad
            pad_len = max_length - len(seq)
            seq = seq + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

            sample: dict[str, torch.Tensor] = {
                "input_ids": torch.tensor(seq, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

            # Labels / targets
            if task_type == "classification":
                assert label_maps is not None
                label_vals = []
                for t in targets:
                    val = row.get(t)
                    if pd.notna(val) and val in label_maps[t]:
                        label_vals.append(label_maps[t][val])
                    else:
                        label_vals.append(-100)
                sample["labels"] = torch.tensor(label_vals, dtype=torch.long)
            else:
                target_vals = [float(row[t]) for t in targets]
                sample["targets"] = torch.tensor(target_vals, dtype=torch.float)

            # Drug one-hot
            if drug_map is not None:
                drug_oh = torch.zeros(len(drug_map), dtype=torch.float)
                drug_val = row.get("Drug")
                if pd.notna(drug_val) and drug_val in drug_map:
                    drug_oh[drug_map[drug_val]] = 1.0
                sample["drug_onehot"] = drug_oh

            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Helpers for building label / drug maps
# ---------------------------------------------------------------------------


def build_label_maps(
    df: pd.DataFrame, targets: list[str]
) -> dict[str, dict[str, int]]:
    """Build {target_name: {label_value: int_index}} from the training split."""
    label_maps: dict[str, dict[str, int]] = {}
    for t in targets:
        unique_vals = sorted(
            [v for v in df[t].dropna().unique()], key=str
        )
        label_maps[t] = {v: i for i, v in enumerate(unique_vals)}
    return label_maps


def build_drug_map(df: pd.DataFrame) -> dict[str, int]:
    """Build {drug_name: int_index} from the training split."""
    unique_drugs = sorted(
        [str(v) for v in df["Drug"].dropna().unique()]
    )
    return {d: i for i, d in enumerate(unique_drugs)}
