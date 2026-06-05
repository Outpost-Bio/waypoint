"""
Metric computation for benchmark tasks.

Classification primary metric: macro-averaged F1 (mean across targets).
Also reports ROC-AUC and PR-AUC (average precision). Multiclass (3+ labels)
uses sklearn macro one-vs-one ROC-AUC; PR-AUC uses the same OVO pairs with
pairwise average precision (sklearn has no ``multi_class`` for AP). Binary
tasks use standard ROC-AUC and AP on the positive class.
Regression primary metric: R² clamped to [0, 1] (mean across targets).
Final benchmark score: mean of all task primary scores.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predictions_to_arrays(
    preds,
    labels,
    task_type: str,
    n_targets: int,
):
    """Convert Trainer predictions to (y_true_list, y_pred_list, y_prob_list)."""
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()

    logits = None
    if task_type == "classification" and preds.ndim == 3:
        logits = preds
        preds = np.argmax(preds, axis=-1)

    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]

    y_true_list, y_pred_list = [], []
    y_prob_list = None
    probs = None
    if logits is not None:
        probs = softmax(logits, axis=-1)
        y_prob_list = []

    for i in range(n_targets):
        if task_type == "classification":
            mask = labels[:, i] != -100
            y_true_list.append(np.asarray(labels[mask, i]))
            y_pred_list.append(np.asarray(preds[mask, i]))
            if y_prob_list is not None and probs is not None:
                y_prob_list.append(np.asarray(probs[mask, i, :]))
        else:
            y_true_list.append(np.asarray(labels[:, i]))
            y_pred_list.append(np.asarray(preds[:, i]))

    return y_true_list, y_pred_list, y_prob_list


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _pr_auc_macro_ovo(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Macro-averaged one-vs-one PR (average precision).

    For each class pair (i, j), restrict to those samples, take binary labels
    vs ``j`` and scores ``P(j) / (P(i)+P(j))``, then average AP over pairs.
    Binary classification uses the positive-class column only.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    if n_classes < 2 or len(y_true) == 0:
        return None
    if n_classes == 2:
        try:
            return float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            return None
    aps: list[float] = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask = (y_true == i) | (y_true == j)
            if np.count_nonzero(mask) < 2:
                continue
            yt = y_true[mask]
            if not (np.any(yt == i) and np.any(yt == j)):
                continue
            y_bin = (yt == j).astype(int)
            p = y_prob[mask]
            denom = p[:, i] + p[:, j]
            score_pos = np.divide(
                p[:, j],
                denom,
                out=np.full_like(p[:, j], 0.5, dtype=float),
                where=denom > 1e-12,
            )
            try:
                ap = average_precision_score(y_bin, score_pos)
            except ValueError:
                continue
            if np.isfinite(ap):
                aps.append(float(ap))
    if not aps:
        return None
    return float(np.mean(aps))


def _roc_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float | None, float | None]:
    """ROC-AUC and PR-AUC (average precision).

    Binary (2 classes): probability of the positive class. Multiclass (3+):
    ``roc_auc_score(..., multi_class="ovo", average="macro")``; PR uses the
    same one-vs-one pairing via :func:`_pr_auc_macro_ovo`.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0 or y_prob.ndim != 2:
        return None, None
    n_classes = y_prob.shape[1]
    if n_classes < 2:
        return None, None
    try:
        if n_classes == 2:
            scores = y_prob[:, 1]
            roc = float(roc_auc_score(y_true, scores))
            pr = float(average_precision_score(y_true, scores))
        else:
            roc = float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovo", average="macro"
                )
            )
            pr = _pr_auc_macro_ovo(y_true, y_prob)
    except ValueError:
        return None, None
    if pr is None or not np.isfinite(roc) or not np.isfinite(pr):
        return None, None
    return roc, pr


def _classification_metrics(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    target_names: list[str],
    y_prob_list: list[np.ndarray] | None = None,
) -> dict:
    metrics: dict = {}
    f1_macros: list[float] = []
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []

    for i, (y_true, y_pred, name) in enumerate(
        zip(y_true_list, y_pred_list, target_names)
    ):
        if len(y_true) == 0:
            continue
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics[f"accuracy_{name}"] = acc
        metrics[f"balanced_accuracy_{name}"] = bal_acc
        metrics[f"f1_macro_{name}"] = f1_m
        f1_macros.append(f1_m)

        if y_prob_list is not None and i < len(y_prob_list):
            y_prob = y_prob_list[i]
            roc, pr = _roc_pr_auc(y_true, y_prob)
            if roc is not None and pr is not None:
                n_classes = y_prob.shape[1]
                if n_classes >= 3:
                    metrics[f"roc_auc_macro_ovo_{name}"] = roc
                    metrics[f"pr_auc_macro_ovo_{name}"] = pr
                else:
                    metrics[f"roc_auc_{name}"] = roc
                    metrics[f"pr_auc_{name}"] = pr
                roc_aucs.append(roc)
                pr_aucs.append(pr)

    if f1_macros:
        metrics["f1_macro_mean"] = float(np.mean(f1_macros))
    if roc_aucs:
        metrics["roc_auc_mean"] = float(np.mean(roc_aucs))
    if pr_aucs:
        metrics["pr_auc_mean"] = float(np.mean(pr_aucs))
    return metrics


def _regression_metrics(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    target_names: list[str],
) -> dict:
    metrics: dict = {}
    r2s: list[float] = []

    for y_true, y_pred, name in zip(y_true_list, y_pred_list, target_names):
        if len(y_true) == 0:
            continue
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[f"mse_{name}"] = mse
        metrics[f"r2_{name}"] = r2
        r2s.append(r2)

        if len(y_true) > 1:
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
            if np.isfinite(pearson):
                metrics[f"pearson_{name}"] = pearson
            sp, _ = spearmanr(y_true, y_pred)
            if np.isfinite(sp):
                metrics[f"spearman_{name}"] = sp

    if r2s:
        metrics["r2_mean"] = float(np.mean(r2s))
    return metrics


def get_metrics(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    target_names: list[str],
    task_type: str,
    y_prob_list: list[np.ndarray] | None = None,
) -> dict:
    """Compute metrics for a task."""
    if task_type == "classification":
        return _classification_metrics(
            y_true_list, y_pred_list, target_names, y_prob_list
        )
    return _regression_metrics(y_true_list, y_pred_list, target_names)


def score_task(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    target_names: list[str],
    task_type: str,
    y_prob_list: list[np.ndarray] | None = None,
) -> tuple[float, dict]:
    """Score a single task. Returns (primary_score, full_metrics)."""
    metrics = get_metrics(
        y_true_list, y_pred_list, target_names, task_type, y_prob_list
    )

    if task_type == "classification":
        score = metrics.get("f1_macro_mean", 0.0)
    else:
        score = max(0.0, metrics.get("r2_mean", 0.0))

    return score, metrics
