"""EE-DrugSeq benchmark scoring utilities.

The functions in this module are intentionally framework-agnostic. They score
target-level perturbation deltas and can be used for baselines, model
submissions, or smoke tests without importing AnnData or PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vcell.utils.metrics import pearson


@dataclass(frozen=True)
class BenchmarkScore:
    """Aggregate expression-response benchmark metrics."""

    model: str
    n_targets: int
    n_genes: int
    des_topk: float
    pds: float
    mae: float
    delta_pearson_mean: float
    delta_spearman_mean: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "n_targets": self.n_targets,
            "n_genes": self.n_genes,
            "des_topk": self.des_topk,
            "pds": self.pds,
            "mae": self.mae,
            "delta_pearson_mean": self.delta_pearson_mean,
            "delta_spearman_mean": self.delta_spearman_mean,
        }


@dataclass(frozen=True)
class ExpressionDeltaMetricConfig:
    """Back-compatible config wrapper for expression-delta scoring."""

    de_top_k: int = 100


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    return arr


def _validate_pair(true_delta: np.ndarray, pred_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = _as_2d(true_delta)
    pred = _as_2d(pred_delta)
    if true.shape != pred.shape:
        raise ValueError(f"true/pred shape mismatch: {true.shape} vs {pred.shape}")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("true_delta and pred_delta must be finite")
    return true, pred


def _safe_nanmean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _rank_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation for two vectors without requiring scipy."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        return float("nan")
    return pearson(_rank_average_ties(a), _rank_average_ties(b))


def topk_abs_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Overlap of top-|effect| genes in one perturbation delta vector."""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        return float("nan")
    k = int(min(max(k, 1), true.size))
    true_top = set(np.argsort(-np.abs(true))[:k])
    pred_top = set(np.argsort(-np.abs(pred))[:k])
    return len(true_top & pred_top) / k


def differential_expression_score(true_delta: np.ndarray, pred_delta: np.ndarray, *, top_k: int = 100) -> float:
    """Mean top-|effect| overlap across perturbations.

    This is a mini-bulk adaptation of the VCC differential-expression idea. It
    uses top absolute deltas rather than unstable per-target p-values.
    """
    true, pred = _validate_pair(true_delta, pred_delta)
    return float(np.mean([topk_abs_overlap(t, p, top_k) for t, p in zip(true, pred)]))


def topk_de_overlap(true_delta: np.ndarray, pred_delta: np.ndarray, k: int = 100) -> float:
    """Back-compatible alias for :func:`differential_expression_score`."""
    return differential_expression_score(true_delta, pred_delta, top_k=k)


def perturbation_discrimination_score(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    *,
    exclude_gene_indices: list[int] | np.ndarray | None = None,
) -> float:
    """Mean rank-based perturbation discrimination score.

    For each predicted perturbation p, all true perturbation deltas are ranked
    by L1 distance to pred[p]. A score of 1 means the matching true perturbation
    is closest; 0 means it is farthest.
    """
    true, pred = _validate_pair(true_delta, pred_delta)
    n_targets, n_genes = true.shape
    if n_targets < 2:
        return float("nan")

    scores = []
    all_genes = np.arange(n_genes)
    excluded = np.asarray(exclude_gene_indices) if exclude_gene_indices is not None else None
    for idx in range(n_targets):
        genes = all_genes
        if excluded is not None:
            genes = all_genes[all_genes != int(excluded[idx])]
        distances = np.abs(true[:, genes] - pred[idx, genes]).sum(axis=1)
        order = np.argsort(distances, kind="mergesort")
        rank_index = int(np.where(order == idx)[0][0])
        scores.append(1.0 - rank_index / (n_targets - 1))
    return float(np.mean(scores))


def expression_benchmark_metrics(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    *,
    model: str,
    top_k: int = 100,
) -> BenchmarkScore:
    """Score target-level expression-delta predictions."""
    true, pred = _validate_pair(true_delta, pred_delta)
    pearsons = [pearson(p, t) for t, p in zip(true, pred)]
    spearmans = [spearman_vector(p, t) for t, p in zip(true, pred)]
    return BenchmarkScore(
        model=model,
        n_targets=int(true.shape[0]),
        n_genes=int(true.shape[1]),
        des_topk=differential_expression_score(true, pred, top_k=top_k),
        pds=perturbation_discrimination_score(true, pred),
        mae=float(np.mean(np.abs(true - pred))),
        delta_pearson_mean=_safe_nanmean(pearsons),
        delta_spearman_mean=_safe_nanmean(spearmans),
    )


def expression_delta_metrics(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    *,
    config: ExpressionDeltaMetricConfig | None = None,
    exclude_gene_indices: list[int] | np.ndarray | None = None,
) -> dict[str, float]:
    """Back-compatible dict-style expression metric API."""
    cfg = config or ExpressionDeltaMetricConfig()
    true, pred = _validate_pair(true_delta, pred_delta)
    pearsons = [pearson(p, t) for t, p in zip(true, pred)]
    spearmans = [spearman_vector(p, t) for t, p in zip(true, pred)]
    diff = pred - true
    return {
        "n_perturbations": float(true.shape[0]),
        "n_genes": float(true.shape[1]),
        "delta_pearson_mean": _safe_nanmean(pearsons),
        "delta_spearman_mean": _safe_nanmean(spearmans),
        "delta_mae": float(np.mean(np.abs(diff))),
        "delta_rmse": float(np.sqrt(np.mean(diff * diff))),
        "topk_de_overlap": differential_expression_score(true, pred, top_k=cfg.de_top_k),
        "pds": perturbation_discrimination_score(true, pred, exclude_gene_indices=exclude_gene_indices),
    }


def zero_delta_baseline(true_delta: np.ndarray) -> np.ndarray:
    """NTC/no-effect baseline prediction."""
    return np.zeros_like(_as_2d(true_delta), dtype=np.float64)


def leave_one_out_mean_baseline(true_delta: np.ndarray) -> np.ndarray:
    """For each target, predict the mean delta of all other targets."""
    true = _as_2d(true_delta)
    n_targets = true.shape[0]
    if n_targets < 2:
        raise ValueError("leave-one-out baseline requires at least two targets")
    return (true.sum(axis=0, keepdims=True) - true) / (n_targets - 1)


def nearest_neighbor_delta_baseline(features: np.ndarray, true_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy the expression delta of the nearest other target in feature space."""
    feat = _as_2d(features)
    true = _as_2d(true_delta)
    if feat.shape[0] != true.shape[0]:
        raise ValueError("features and true_delta must have the same number of targets")
    distances = np.sqrt(((feat[:, None, :] - feat[None, :, :]) ** 2).mean(axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbor = np.argmin(distances, axis=1)
    return true[neighbor].copy(), neighbor


def classification_metrics(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> dict[str, float]:
    """Macro-F1 and balanced accuracy for MoA/toxicity state recovery."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }