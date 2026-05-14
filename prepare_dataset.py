#!/usr/bin/env python3
"""
Convert a taxonomic abundance matrix into a serialized waypoint-format dataset.

The output is a parquet file with two list-columns, ``Taxa`` and
``Relative Abundances``, indexed by sample ID. It can then be fed to any
waypoint script that accepts a local dataset, e.g.:

    python pretrain.py --data my_dataset.parquet ...
    python embed.py    --data my_dataset.parquet ...

Usage:
    # MGnify-style TSV (taxa as rows, samples as columns, first column = lineage)
    python prepare_dataset.py \\
        --input examples/abundance_matrix.tsv \\
        --output examples/abundance_matrix.parquet

    # Phyloseq-style CSV (samples as rows, taxa as columns)
    python prepare_dataset.py \\
        --input my_matrix.csv \\
        --output my_dataset.parquet \\
        --orientation samples_as_rows

    # Bare genus names instead of full lineage strings
    python prepare_dataset.py \\
        --input genus_matrix.csv --output genus.parquet \\
        --taxonomy_format genus

    # Attach per-sample metadata (labels) for benchmarking-style use
    python prepare_dataset.py \\
        --input my_matrix.csv --output my_dataset.parquet \\
        --metadata sample_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.abundance_matrix import load_abundance_matrix, matrix_to_waypoint_df


def _load_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    if df.index.name is None:
        df = df.set_index(df.columns[0])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert an abundance matrix into a serialized waypoint dataset",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the abundance matrix (.csv / .tsv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (.parquet recommended; .csv also supported).",
    )
    parser.add_argument(
        "--orientation",
        default="auto",
        choices=["auto", "samples_as_rows", "taxa_as_rows"],
        help=(
            "Matrix layout. 'samples_as_rows': rows = samples, columns = taxa. "
            "'taxa_as_rows': rows = taxa, columns = samples (first column is the "
            "taxonomy lineage; MGnify-style). 'auto' (default): detected from "
            "the first column header."
        ),
    )
    parser.add_argument(
        "--taxonomy_format",
        default="full",
        help=(
            "'full' (default) if taxa identifiers are full lineage strings "
            "(k__...;p__...;g__...), or a rank name ('species', 'genus', "
            "'family', 'order', 'class', 'phylum', 'kingdom') to prefix bare "
            "names with the rank tag. Bare-name mode disables higher-rank "
            "fallback in the tokenizer."
        ),
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip row-normalization to relative abundances (use if your matrix already holds relative abundances).",
    )
    parser.add_argument(
        "--keep_zeros",
        action="store_true",
        help="Keep zero-abundance entries in each sample's lists (default: drop them).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV/TSV/parquet of per-sample metadata to merge in as extra columns (indexed by sample ID).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading abundance matrix from {input_path} ...")
    matrix = load_abundance_matrix(input_path, orientation=args.orientation)
    print(f"  shape: {matrix.shape[0]} samples x {matrix.shape[1]} taxa")

    metadata = None
    if args.metadata is not None:
        print(f"Loading metadata from {args.metadata} ...")
        metadata = _load_metadata(Path(args.metadata))
        print(f"  columns: {list(metadata.columns)}")

    print("Converting to waypoint format ...")
    df = matrix_to_waypoint_df(
        matrix,
        taxonomy_format=args.taxonomy_format,
        normalize=not args.no_normalize,
        drop_zeros=not args.keep_zeros,
        metadata=metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path)
    elif suffix in (".csv", ".tsv", ".tab"):
        sep = "\t" if suffix in (".tsv", ".tab") else ","
        df.to_csv(output_path, sep=sep)
    else:
        raise ValueError(
            f"Unsupported output extension: {suffix!r}. Use .parquet, .csv, or .tsv."
        )
    print(f"Saved {len(df):,} samples to {output_path}")


if __name__ == "__main__":
    main()
