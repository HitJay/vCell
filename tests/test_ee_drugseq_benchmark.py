from __future__ import annotations

import numpy as np

from vcell.benchmark.ee_drugseq import (
    ExpressionDeltaMetricConfig,
    classification_metrics,
    expression_delta_metrics,
    perturbation_discrimination_score,
    topk_de_overlap,
)


def test_topk_de_overlap_perfect_and_partial():
    true = np.array([[5.0, -4.0, 0.1, 0.0], [0.0, 3.0, -2.0, 0.2]])
    pred = true.copy()
    assert topk_de_overlap(true, pred, k=2) == 1.0
    pred2 = np.array([[5.0, 0.0, 4.0, 0.1], [0.0, 3.0, 0.2, -2.0]])
    assert topk_de_overlap(true, pred2, k=2) == 0.5


def test_perturbation_discrimination_score_perfect():
    true = np.eye(4)
    pred = np.eye(4)
    assert perturbation_discrimination_score(true, pred) == 1.0


def test_perturbation_discrimination_score_worst_for_reversed():
    true = np.eye(4)
    pred = true[::-1]
    score = perturbation_discrimination_score(true, pred)
    assert 0.0 <= score < 0.5


def test_expression_delta_metrics_keys():
    true = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    pred = true.copy()
    metrics = expression_delta_metrics(true, pred, config=ExpressionDeltaMetricConfig(de_top_k=2))
    assert metrics["delta_pearson_mean"] > 0.999
    assert metrics["delta_spearman_mean"] > 0.999
    assert metrics["delta_mae"] == 0.0
    assert metrics["topk_de_overlap"] == 1.0
    assert metrics["pds"] == 1.0


def test_classification_metrics():
    metrics = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert 0.0 < metrics["balanced_accuracy"] < 1.0