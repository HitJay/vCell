from __future__ import annotations

import numpy as np

from vcell.benchmark import (
    differential_expression_score,
    expression_benchmark_metrics,
    leave_one_out_mean_baseline,
    nearest_neighbor_delta_baseline,
    perturbation_discrimination_score,
    spearman_vector,
    topk_abs_overlap,
    zero_delta_baseline,
)


def test_topk_abs_overlap_perfect_and_partial():
    y_true = np.array([5.0, -4.0, 0.2, 0.1])
    assert topk_abs_overlap(y_true, y_true, k=2) == 1.0
    y_pred = np.array([0.1, 8.0, 7.0, 0.0])
    assert topk_abs_overlap(y_true, y_pred, k=2) == 0.5


def test_expression_metrics_perfect_prediction():
    true = np.array([[1.0, 0.0, -2.0], [0.0, 3.0, 1.0], [-1.0, 0.5, 0.0]])
    score = expression_benchmark_metrics(true, true, model="oracle", top_k=2)
    assert score.des_topk == 1.0
    assert score.pds == 1.0
    assert score.mae == 0.0
    assert score.delta_pearson_mean > 0.999


def test_perturbation_discrimination_penalizes_swapped_predictions():
    true = np.eye(3)
    swapped = true[[1, 0, 2]]
    assert perturbation_discrimination_score(true, true) == 1.0
    assert perturbation_discrimination_score(true, swapped) < 1.0


def test_baselines_shapes_and_nearest_neighbor():
    true = np.array([[1.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
    zero = zero_delta_baseline(true)
    mean = leave_one_out_mean_baseline(true)
    assert zero.shape == true.shape
    assert mean.shape == true.shape
    features = np.array([[0.0], [0.1], [10.0]])
    pred, neighbor = nearest_neighbor_delta_baseline(features, true)
    assert neighbor.tolist() == [1, 0, 1]
    assert np.allclose(pred[0], true[1])


def test_spearman_vector_handles_ties():
    assert spearman_vector(np.array([1, 2, 3]), np.array([1, 2, 3])) > 0.999
    assert spearman_vector(np.array([1, 1, 2]), np.array([1, 1, 2])) > 0.999


def test_differential_expression_score_averages_targets():
    true = np.array([[5.0, 4.0, 0.0], [0.0, 3.0, 2.0]])
    pred = np.array([[5.0, 4.0, 0.0], [4.0, 0.0, 2.0]])
    assert differential_expression_score(true, pred, top_k=2) == 0.75