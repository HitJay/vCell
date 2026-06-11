"""Smoke tests for the leakage-aware vAssay evaluation framework."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcell.vassay import (
    cv_splits,
    load_vassay_csv,
    regression_metrics,
    run_cv,
    baseline_metrics,
)


@pytest.fixture()
def mini_csv(tmp_path):
    rng = np.random.default_rng(0)
    plates = ["P1", "P2", "P3", "P4"]
    treatments = ["DMSO", "BAM15", "siNTC-1", "siATP5B", "siMFN2"]
    rows = []
    for pi, pl in enumerate(plates):
        for wi in range(10):
            tr = treatments[wi % len(treatments)]
            # plate-correlated signal to expose leakage between schemes
            base = 100 + pi * 5
            rows.append({
                "Plate": pl, "ImgID": f"{pl}_{wi}", "Treatment": tr,
                "AUC-avg-con.": base + rng.normal(0, 3),
                "Ave-Max/Basal": 2.0 + rng.normal(0, 0.1),
                **{f"DINO_Feature_{i}": rng.normal(pi, 1) for i in range(20)},
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "train_C1.csv"
    df.to_csv(p, index=False)
    return p


def test_load_and_targets(mini_csv):
    d = load_vassay_csv(mini_csv, target="AUC")
    assert d.n == 40
    assert d.X.shape == (40, 20)
    assert d.target == "AUC-avg-con."
    assert d.channel == "C1"
    # siRNA vs compound split
    assert d.is_sirna.sum() == 24  # 3 of 5 treatments are siRNA, evenly spread
    assert (~d.is_sirna).sum() == 16


def test_metrics_perfect():
    y = np.array([1.0, 2, 3, 4, 5])
    m = regression_metrics(y, y)
    assert m["r2"] > 0.999
    assert m["pearson"] > 0.999


def test_group_plate_has_no_shared_plate(mini_csv):
    d = load_vassay_csv(mini_csv, target="AUC")
    for tr, te in cv_splits(d, "group_plate", n_splits=4):
        assert set(d.plate[tr]).isdisjoint(set(d.plate[te]))


def test_group_treatment_has_no_shared_treatment(mini_csv):
    d = load_vassay_csv(mini_csv, target="AUC")
    for tr, te in cv_splits(d, "group_treatment", n_splits=5):
        assert set(d.treatment[tr]).isdisjoint(set(d.treatment[te]))


def test_run_cv_and_baseline(mini_csv):
    d = load_vassay_csv(mini_csv, target="AUC")
    res = run_cv(d, "ridge", "group_plate")
    assert "r2" in res.metrics_mean
    assert res.oof_pred is not None and not np.isnan(res.oof_pred).all()
    base = baseline_metrics(d, "group_plate")
    assert "r2" in base


def test_aggregate_collapses_replicates(mini_csv):
    # raw: 40 wells; aggregated: one row per (plate, treatment) = 4 plates x 5 trt
    raw = load_vassay_csv(mini_csv, target="AUC", aggregate=False)
    agg = load_vassay_csv(mini_csv, target="AUC", aggregate=True)
    assert raw.n == 40
    assert agg.n == 20  # 4 plates x 5 treatments, each replicate collapsed
    # every aggregated row is a unique (plate, treatment)
    keys = list(zip(agg.plate, agg.treatment))
    assert len(keys) == len(set(keys))


def test_sirna_only_drops_compounds(mini_csv):
    d = load_vassay_csv(mini_csv, target="AUC", sirna_only=True)
    assert set(d.treatment).isdisjoint({"DMSO", "BAM15"})
    assert all(t.startswith("si") for t in d.treatment)
    d2 = load_vassay_csv(mini_csv, target="AUC", sirna_only=True, drop_controls=True)
    assert not any(t.startswith("siNTC") for t in d2.treatment)
