"""Utilities shared by the two webinar demo notebooks.

Both ``webinar_finetune_regression.ipynb`` and ``webinar_finetune_classification.ipynb``
import from here: constants, path/S3/loading helpers, PCA + t-SNE projection,
linear-probe baselines, and plotting utilities. Task-specific config (dataset
name, target column, covariate, S3 URI) lives in each notebook's setup cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "outpost-bio/Waypoint-6m"
COMPASS_REPO = "outpost-bio/Compass"
POOLING = "last_token"
MAX_LENGTH = 512
BATCH_SIZE = 32
SEED = 42

# Vaginal → blue, C-section → red. Covers common spelling variants in the roswall
# dataset so whichever the Hub actually uses is picked up.
DELIVERY_MODE_COLORS = {
    "Vaginal": "royalblue", "vaginal": "royalblue", "V": "royalblue",
    "VD": "royalblue", "Vaginal Delivery": "royalblue",
    "Cesarean": "crimson", "cesarean": "crimson",
    "Caesarean": "crimson", "caesarean": "crimson",
    "C-section": "crimson", "c-section": "crimson",
    "C-Section": "crimson", "CS": "crimson",
    "Caesarean Section": "crimson", "Cesarean Section": "crimson",
}


# ---------------------------------------------------------------------------
# Paths + artifact I/O
# ---------------------------------------------------------------------------

def paths_for(artifact_dir: Path, tag: str) -> dict[str, Path]:
    """Return every on-disk path used for a task's artifacts.

    Directory layout::

        <artifact_dir>/<tag>/
            samples.parquet                    # --data input for embed / finetune
            base_embeddings.parquet            # `waypoint embed` output (base)
            finetuned_embeddings.parquet       # `waypoint embed` output (fine-tuned)
            finetune_run/                      # `waypoint finetune` output_dir
                best_model/                    # the fine-tuned checkpoint
            metrics.json                       # money-slide numbers
            timings.json                       # wall-clock per step
    """
    base = artifact_dir / tag
    base.mkdir(exist_ok=True, parents=True)
    return {
        "samples":    base / "samples.parquet",
        "base_emb":   base / "base_embeddings.parquet",
        "ft_emb":     base / "finetuned_embeddings.parquet",
        "ft_dir":     base / "finetune_run",
        "ft_model":   base / "finetune_run" / "best_model",
        "metrics":    base / "metrics.json",
        "timings":    base / "timings.json",
    }


def load_embeddings(path: Path) -> np.ndarray:
    """Read a ``waypoint embed``-produced parquet and return just the dim_* columns."""
    return pd.read_parquet(path).filter(regex=r"^dim_\d+$").to_numpy()


# ---------------------------------------------------------------------------
# Projection + linear-probe baselines
# ---------------------------------------------------------------------------

def project(emb: np.ndarray, seed: int = 0, include_3d: bool = True) -> dict:
    """PCA (2D + 3D) and t-SNE (2D + optionally 3D).

    3D t-SNE roughly doubles the projection cost — set include_3d=False to skip it.
    """
    pca_full = PCA(n_components=min(50, emb.shape[1], emb.shape[0]), random_state=seed)
    pca_hi = pca_full.fit_transform(emb)
    perp = float(min(30, max(5, (len(emb) - 1) // 3)))
    tsne2 = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=seed).fit_transform(pca_hi)
    out = {
        "pca": pca_hi[:, :2],
        "tsne": tsne2,
        "pca_var": pca_full.explained_variance_ratio_[:2].sum(),
    }
    if include_3d and pca_hi.shape[1] >= 3:
        tsne3 = TSNE(n_components=3, perplexity=perp, init="pca",
                     learning_rate="auto", random_state=seed).fit_transform(pca_hi)
        out["pca3d"] = pca_hi[:, :3]
        out["tsne3d"] = tsne3
        out["pca3d_var"] = float(pca_full.explained_variance_ratio_[:3].sum())
    return out


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, n // 5)
    return idx[n_test:], idx[:n_test]


def ridge_score(X: np.ndarray, y: np.ndarray, seed: int = SEED,
                cov: np.ndarray | None = None) -> dict:
    """RidgeCV on frozen embeddings, held-out 20% random split.

    Pass ``cov`` to concatenate a one-hot covariate onto the embeddings — mirrors
    the fine-tune head, which sees ``[encoder features | covariate]``. Alpha is
    auto-tuned via CV over the passed grid.
    """
    train_idx, test_idx = _split_indices(len(X), seed)
    scaler = StandardScaler().fit(X[train_idx])
    Xtr = scaler.transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])
    if cov is not None:
        Xtr = np.hstack([Xtr, cov[train_idx]])
        Xte = np.hstack([Xte, cov[test_idx]])
    reg = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)).fit(Xtr, y[train_idx])
    pred = reg.predict(Xte)
    return {
        "r2":  float(r2_score(y[test_idx], pred)),
        "mae": float(mean_absolute_error(y[test_idx], pred)),
        "alpha": float(reg.alpha_),
        "y_true": y[test_idx], "y_pred": pred,
    }


def logreg_score(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """Class-balanced logistic regression on frozen embeddings, 80/20 random split."""
    train_idx, test_idx = _split_indices(len(X), seed)
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=seed).fit(Xtr, y[train_idx])
    pred = clf.predict(Xte)
    out = {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "f1_macro": float(f1_score(y[test_idx], pred, average="macro")),
        "y_true": y[test_idx], "y_pred": pred, "classes": clf.classes_.tolist(),
    }
    if len(clf.classes_) == 2:
        prob = clf.predict_proba(Xte)[:, 1]
        out["roc_auc"] = float(roc_auc_score((y[test_idx] == clf.classes_[1]).astype(int), prob))
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _categorical_palette(n: int) -> list[str]:
    """Return n visually distinct colors — Alphabet (26) covers most drug screens."""
    alphabet = px.colors.qualitative.Alphabet
    if n <= len(alphabet):
        return alphabet[:n]
    return px.colors.sample_colorscale("hsv", [i / n for i in range(n)])


def _bucket_top_n(series: pd.Series, top_n: int, other_label: str = "other"):
    """Keep top-N categories by count; bucket the rest under ``other_label``."""
    counts = series.value_counts()
    top = counts.head(top_n).index.astype(str).tolist()
    s = series.astype(str)
    bucketed = s.where(s.isin(top), other=other_label)
    n_other = int((bucketed == other_label).sum())
    order = top + ([other_label] if n_other else [])
    palette = _categorical_palette(len(top))
    color_map = {c: palette[i] for i, c in enumerate(top)}
    if n_other:
        color_map[other_label] = "lightgray"
    return bucketed, order, color_map


def plot_regression_stage(proj: dict, df: pd.DataFrame, target: str, covariate: str,
                          title: str, top_n_categories: int = 12):
    """2x2 overview: top row colored by top-N covariate values, bottom by continuous target."""
    bucketed, cat_order, color_map = _bucket_top_n(df[covariate], top_n_categories)
    n_total = df[covariate].nunique()
    other_note = ""
    if len(color_map) < n_total + 1:
        n_other_cats = n_total - top_n_categories
        other_note = f" (top {top_n_categories} of {n_total}; {n_other_cats} others bucketed)"

    frame = pd.DataFrame({
        "PCA1": proj["pca"][:, 0], "PCA2": proj["pca"][:, 1],
        "tSNE1": proj["tsne"][:, 0], "tSNE2": proj["tsne"][:, 1],
        covariate: bucketed.values,
        target: df[target].values,
    })
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"PCA (var explained: {proj['pca_var']:.2f}) — colored by {covariate}{other_note}",
            f"t-SNE — colored by {covariate}",
            f"PCA — colored by {target}",
            f"t-SNE — colored by {target}",
        ),
    )
    seen: set[str] = set()
    for src_x, src_y, col_idx in (("PCA1", "PCA2", 1), ("tSNE1", "tSNE2", 2)):
        cat_fig = px.scatter(frame, x=src_x, y=src_y, color=covariate, opacity=0.75,
                             color_discrete_map=color_map,
                             category_orders={covariate: cat_order})
        for tr in cat_fig.data:
            tr.marker.size = 4 if tr.name == "other" else 6
            tr.showlegend = tr.name not in seen
            seen.add(tr.name)
            tr.legendgroup = tr.name
            fig.add_trace(tr, row=1, col=col_idx)
    for src_x, src_y, col_idx in (("PCA1", "PCA2", 1), ("tSNE1", "tSNE2", 2)):
        cont = px.scatter(frame, x=src_x, y=src_y, color=target,
                          color_continuous_scale="Viridis", opacity=0.85)
        for tr in cont.data:
            tr.marker.size = 6
            tr.showlegend = False
            fig.add_trace(tr, row=2, col=col_idx)
    fig.update_layout(title=title, height=760, legend_title=covariate,
                      coloraxis=dict(colorscale="Viridis",
                                     colorbar=dict(title=target, y=0.23, len=0.45)))
    return fig


def plot_classification_stage(proj: dict, df: pd.DataFrame, target: str, title: str,
                              top_n_categories: int = 12,
                              color_map: dict[str, str] | None = None):
    """1x2 layout: PCA and t-SNE colored by the categorical target.

    Pass ``color_map`` to override auto colors for specific class names.
    """
    bucketed, cat_order, auto_map = _bucket_top_n(df[target], top_n_categories)
    if color_map:
        auto_map = {**auto_map, **color_map}
    frame = pd.DataFrame({
        "PCA1": proj["pca"][:, 0], "PCA2": proj["pca"][:, 1],
        "tSNE1": proj["tsne"][:, 0], "tSNE2": proj["tsne"][:, 1],
        target: bucketed.values,
    })
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"PCA (var explained: {proj['pca_var']:.2f}) — colored by {target}",
            f"t-SNE — colored by {target}",
        ),
    )
    seen: set[str] = set()
    for src_x, src_y, col_idx in (("PCA1", "PCA2", 1), ("tSNE1", "tSNE2", 2)):
        sub = px.scatter(frame, x=src_x, y=src_y, color=target, opacity=0.8,
                         color_discrete_map=auto_map,
                         category_orders={target: cat_order})
        for tr in sub.data:
            tr.marker.size = 5 if tr.name == "other" else 7
            tr.showlegend = tr.name not in seen
            seen.add(tr.name)
            tr.legendgroup = tr.name
            fig.add_trace(tr, row=1, col=col_idx)
    fig.update_layout(title=title, height=460, legend_title=target)
    return fig


def _require_3d(proj: dict) -> None:
    if "pca3d" not in proj or "tsne3d" not in proj:
        raise KeyError(
            "proj is missing 'pca3d'/'tsne3d'. Recompute with project(..., include_3d=True)."
        )


def plot_regression_stage_3d(proj: dict, df: pd.DataFrame, target: str, title: str):
    """3D PCA + 3D t-SNE side by side, colored by the continuous target."""
    _require_3d(proj)
    target_vals = df[target].to_numpy()
    vmin, vmax = float(np.nanmin(target_vals)), float(np.nanmax(target_vals))
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"PCA 3D (var explained: {proj.get('pca3d_var', 0):.2f}) — colored by {target}",
            f"t-SNE 3D — colored by {target}",
        ),
        horizontal_spacing=0.02,
    )
    for key, col_idx in (("pca3d", 1), ("tsne3d", 2)):
        coords = proj[key]
        show_scale = (col_idx == 2)
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=3, color=target_vals,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                opacity=0.75, showscale=show_scale,
                colorbar=dict(title=target, x=1.02) if show_scale else None,
            ),
            showlegend=False,
            hovertemplate=f"{target}: %{{marker.color:.3f}}<extra></extra>",
        ), row=1, col=col_idx)
    fig.update_layout(title=title, height=620, showlegend=False)
    return fig


def plot_classification_stage_3d(proj: dict, df: pd.DataFrame, target: str, title: str,
                                 top_n_categories: int = 12,
                                 color_map: dict[str, str] | None = None):
    """3D PCA + 3D t-SNE side by side, colored by the categorical target."""
    _require_3d(proj)
    bucketed, cat_order, auto_map = _bucket_top_n(df[target], top_n_categories)
    if color_map:
        auto_map = {**auto_map, **color_map}
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"PCA 3D (var explained: {proj.get('pca3d_var', 0):.2f}) — colored by {target}",
            f"t-SNE 3D — colored by {target}",
        ),
        horizontal_spacing=0.02,
    )
    seen: set[str] = set()
    for key, col_idx in (("pca3d", 1), ("tsne3d", 2)):
        coords = proj[key]
        for cat in cat_order:
            mask = bucketed.values == cat
            if not mask.any():
                continue
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers",
                marker=dict(size=3 if cat == "other" else 4,
                            color=auto_map.get(cat, "gray"), opacity=0.7),
                showlegend=(cat not in seen),
                legendgroup=cat, name=cat,
            ), row=1, col=col_idx)
            seen.add(cat)
    fig.update_layout(title=title, height=620, legend_title=target)
    return fig


def plot_per_covariate_facets(proj: dict, df: pd.DataFrame, target: str, covariate: str,
                              title: str, top_n: int = 12):
    """Small-multiples: one row per top-N covariate value, cols = [PCA, t-SNE].

    Each panel greys-out every sample and highlights that value's samples colored
    by the continuous target. Uses ``go.Scattergl`` so 15k-point backgrounds stay snappy.
    """
    counts = df[covariate].value_counts()
    top = counts.head(top_n).index.tolist()
    n_total = df[covariate].nunique()
    note = f" (top {top_n} of {n_total})" if top_n < n_total else ""

    pca = proj["pca"]
    tsne = proj["tsne"]
    target_vals = df[target].to_numpy()
    cov_vals = df[covariate].to_numpy()
    vmin = float(np.nanmin(target_vals))
    vmax = float(np.nanmax(target_vals))

    subplot_titles = []
    for name in top:
        n_pts = int((cov_vals == name).sum())
        subplot_titles.append(f"{name} — PCA (n={n_pts})")
        subplot_titles.append(f"{name} — t-SNE")

    fig = make_subplots(
        rows=len(top), cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06, vertical_spacing=0.02,
    )
    for i, name in enumerate(top, start=1):
        mask = cov_vals == name
        for col_idx, coords in ((1, pca), (2, tsne)):
            fig.add_trace(go.Scattergl(
                x=coords[~mask, 0], y=coords[~mask, 1],
                mode="markers",
                marker=dict(size=2, color="lightgray", opacity=0.25),
                showlegend=False, hoverinfo="skip",
            ), row=i, col=col_idx)
            show_scale = (i == 1 and col_idx == 2)
            fig.add_trace(go.Scattergl(
                x=coords[mask, 0], y=coords[mask, 1],
                mode="markers",
                marker=dict(
                    size=6, color=target_vals[mask],
                    colorscale="Viridis", cmin=vmin, cmax=vmax,
                    showscale=show_scale,
                    colorbar=dict(title=target, len=0.9) if show_scale else None,
                    line=dict(width=0.3, color="black"),
                ),
                showlegend=False, name=str(name),
                hovertemplate=f"<b>{name}</b><br>{target}: %{{marker.color:.3f}}<extra></extra>",
            ), row=i, col=col_idx)
    fig.update_layout(title=title + note, height=260 * len(top), showlegend=False)
    return fig
