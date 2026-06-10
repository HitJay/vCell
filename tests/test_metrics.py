import numpy as np

from vcell.utils.metrics import delta_metrics, mse, pearson, r2_score


def test_mse_zero_for_identical():
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_r2_perfect():
    assert abs(r2_score([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9


def test_pearson_perfect_positive():
    assert abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9


def test_pearson_perfect_negative():
    assert abs(pearson([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9


def test_pearson_constant_is_nan():
    assert np.isnan(pearson([1, 1, 1], [1, 2, 3]))


def test_delta_metrics_keys_and_values():
    m = delta_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert set(m) == {"delta_pearson", "delta_r2", "delta_mse"}
    assert abs(m["delta_r2"] - 1.0) < 1e-9
    assert m["delta_mse"] == 0.0
