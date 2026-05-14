"""
Convert taxonomic abundance matrices into the waypoint DataFrame format.

The canonical waypoint format is a DataFrame with two list-columns:
    Taxa                 : list[str] of taxonomy strings, one per non-zero taxon
    Relative Abundances  : list[float] of matching abundances

This module turns a sample x taxa matrix (rows = samples, columns = taxa)
into that format so it can feed ``MicrobiomePretrainingDataset`` /
``MicrobiomeBenchmarkDataset`` directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tokenizer import TAXON_RANK_PREFIXES


def _format_taxon_columns(columns, taxonomy_format: str) -> list[str]:
    if taxonomy_format == "full":
        return [str(c).strip() for c in columns]
    if taxonomy_format not in TAXON_RANK_PREFIXES:
        raise ValueError(
            f"taxonomy_format must be 'full' or one of "
            f"{list(TAXON_RANK_PREFIXES.keys())}, got: {taxonomy_format!r}"
        )
    prefix = TAXON_RANK_PREFIXES[taxonomy_format]
    formatted = []
    for c in columns:
        c_str = str(c).strip()
        formatted.append(c_str if c_str.startswith(prefix) else f"{prefix}{c_str}")
    return formatted


def matrix_to_waypoint_df(
    matrix: pd.DataFrame,
    *,
    taxonomy_format: str = "full",
    normalize: bool = True,
    drop_zeros: bool = True,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert a sample x taxa abundance matrix into waypoint format.

    Parameters
    ----------
    matrix:
        Index = sample IDs, columns = taxon names, values = abundance.
    taxonomy_format:
        ``"full"`` (default): column names are already full taxonomy strings
        (e.g. ``"k__Bacteria;p__Firmicutes;g__Lactobacillus"``).
        A rank name (``"species"``, ``"genus"``, ``"family"``, ``"order"``,
        ``"class"``, ``"phylum"``, ``"kingdom"``): column names are bare taxon
        names; each is prefixed with the rank tag (e.g. ``g__`` for genus).
        Note that with bare names the tokenizer cannot fall back to a higher
        rank, so the rank used here must match the model's tokenization rank.
    normalize:
        Row-normalize so each sample's abundances sum to 1.
    drop_zeros:
        Omit zero-abundance entries from each sample's lists. Recommended.
    metadata:
        Optional DataFrame indexed by sample ID; merged in as extra columns
        (e.g. labels for downstream tasks).
    """
    if matrix.empty:
        raise ValueError("matrix is empty")

    abundances = matrix.astype(float)
    taxon_strings = _format_taxon_columns(matrix.columns, taxonomy_format)

    if normalize:
        row_sums = abundances.sum(axis=1)
        zero_rows = row_sums == 0
        if zero_rows.any():
            n = int(zero_rows.sum())
            sample_ids = abundances.index[zero_rows].tolist()
            preview = sample_ids[:5]
            suffix = "..." if n > 5 else ""
            raise ValueError(
                f"{n} sample(s) have zero total abundance and cannot be "
                f"normalized: {preview}{suffix}"
            )
        abundances = abundances.div(row_sums, axis=0)

    taxa_lists: list[list[str]] = []
    ra_lists: list[list[float]] = []
    for _, row in abundances.iterrows():
        values = row.to_numpy()
        if drop_zeros:
            keep = values > 0
            taxa_lists.append([taxon_strings[i] for i, k in enumerate(keep) if k])
            ra_lists.append([float(v) for v, k in zip(values, keep) if k])
        else:
            taxa_lists.append(list(taxon_strings))
            ra_lists.append([float(v) for v in values])

    out = pd.DataFrame(
        {"Taxa": taxa_lists, "Relative Abundances": ra_lists},
        index=abundances.index,
    )
    out.index.name = matrix.index.name or "sample_id"

    if metadata is not None:
        out = out.join(metadata, how="left")

    return out


_TAXONOMY_HEADER_NAMES = {"taxonomy", "lineage", "taxon", "otu", "#otu id"}


def load_abundance_matrix(
    path: str | Path,
    *,
    orientation: str = "auto",
    sample_id_col: str | int | None = 0,
    sep: str | None = None,
) -> pd.DataFrame:
    """Load an abundance matrix from CSV/TSV and return it as samples x taxa.

    Parameters
    ----------
    orientation:
        ``"samples_as_rows"`` (rows = samples, columns = taxa) — the default
        layout used by :func:`matrix_to_waypoint_df`.
        ``"taxa_as_rows"`` (rows = taxa, columns = samples) — MGnify-style
        files where the first column is a ``taxonomy`` lineage string and the
        remaining columns are sample IDs. Transposed on load.
        ``"auto"`` (default): detect by checking whether the first column
        header (case-insensitive) matches one of
        ``{"taxonomy", "lineage", "taxon", "otu", "#otu id"}``; if so, treat
        as ``taxa_as_rows``, otherwise ``samples_as_rows``.
    sample_id_col:
        Used only when ``orientation`` resolves to ``"samples_as_rows"``: the
        column to use as the sample-ID index. Default ``0`` (first column);
        pass ``None`` to leave the default integer index.
    sep:
        CSV delimiter. Auto-detected from the file extension if omitted.
    """
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)

    if orientation == "auto":
        first_col = str(df.columns[0]).strip().lower()
        orientation = (
            "taxa_as_rows"
            if first_col in _TAXONOMY_HEADER_NAMES
            else "samples_as_rows"
        )

    if orientation == "taxa_as_rows":
        taxonomy_col = df.columns[0]
        df = df.set_index(taxonomy_col).T
        df.index.name = "sample_id"
        return df

    if orientation == "samples_as_rows":
        if sample_id_col is not None:
            id_col = (
                df.columns[sample_id_col]
                if isinstance(sample_id_col, int)
                else sample_id_col
            )
            df = df.set_index(id_col)
        return df

    raise ValueError(
        f"orientation must be one of 'auto', 'samples_as_rows', "
        f"'taxa_as_rows'; got: {orientation!r}"
    )
