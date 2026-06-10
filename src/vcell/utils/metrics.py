"""Evaluation metrics (NumPy, framework-agnostic).

The headline virtual-cell metric is the agreement between *predicted* and
*observed* perturbation effects, measured on the mean Δ-expression vector
(perturbed mean minus control mean) across genes.
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error over all elements."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination over the flattened inputs."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation coefficient between two flattened vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def delta_metrics(
    pred_delta: ArrayLike,
    true_delta: ArrayLike,
) -> dict[str, float]:
    """Pearson / R² / MSE between predicted and observed Δ-expression vectors."""
    return {
        "delta_pearson": pearson(pred_delta, true_delta),
        "delta_r2": r2_score(true_delta, pred_delta),
        "delta_mse": mse(true_delta, pred_delta),
    }
