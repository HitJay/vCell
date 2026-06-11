"""vAssay model-evaluation framework (leakage-aware).

The legacy vAssay TabPFN models were trained/validated with a *random* shuffled
KFold, which lets wells from the same plate (and same treatment) fall into both
train and test — inflating the reported R² (~0.77). This module re-evaluates the
imaging → Seahorse models under **leakage-aware grouping** (GroupKFold by plate
or by treatment / LeaveOneGroupOut) so the reported numbers reflect real
generalization to unseen plates / unseen perturbations.

It also exposes a uniform interface for the channel × target × model × cv-scheme
benchmark used in :mod:`scripts.vassay_benchmark`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
TARGET_COLS = ("AUC-avg-con.", "Ave-Max-avg", "Ave-Basal-avg",
               "Spare-avg", "Ave-Max/Basal")
#: short alias -> column name in the training csv
TARGET_ALIASES = {
    "AUC": "AUC-avg-con.",
    "Max": "Ave-Max-avg",
    "Basal": "Ave-Basal-avg",
    "Spare": "Spare-avg",
    "MB": "Ave-Max/Basal",
}
META_COLS = ("Plate", "ImgID", "Treatment")
#: control/compound treatments (not gene knockdowns) — used to split the data
#: into "compound" vs "siRNA" domains for the domain-shift analysis.
NON_SIRNA_TREATMENTS = {"No add", "DMSO", "BAM15", "CCCP", "FCCP", "Smol"}


@dataclass
class VassayData:
    X: np.ndarray
    y: np.ndarray
    plate: np.ndarray
    treatment: np.ndarray
    img_id: np.ndarray
    feature_names: list[str]
    target: str
    channel: str

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def is_sirna(self) -> np.ndarray:
        return ~np.isin(self.treatment, list(NON_SIRNA_TREATMENTS))


def load_vassay_csv(
    path: str | Path,
    target: str = "AUC",
    aggregate: bool = False,
    sirna_only: bool = False,
    drop_controls: bool = False,
) -> VassayData:
    """Load a ``train_C*.csv`` and pull out features + one Seahorse target.

    Parameters
    ----------
    aggregate
        If True, collapse the (Plate, Treatment) field/well replicates that
        share an identical Seahorse value into a *single* independent unit
        (mean of the DINOv2 features). This removes the label-leakage that
        inflated the random-CV R² (264 image rows → 88 y values → 3 rows share
        each Seahorse measurement). After aggregation every row is an
        independent (plate, treatment) measurement.
    sirna_only
        Keep only siRNA-knockdown treatments (drop compounds) — match the
        deployment domain.
    drop_controls
        Also drop the non-targeting siNTC controls (keep only real targets).
    """
    df = pd.read_csv(path)
    col = TARGET_ALIASES.get(target, target)
    if col not in df.columns:
        raise KeyError(f"target {target!r} -> {col!r} not in {Path(path).name}")
    feat = [c for c in df.columns if c.startswith("DINO_Feature_")]
    feat = sorted(feat, key=lambda c: int(c.split("_")[-1]))
    df = df[df[col].notna()].copy()

    if sirna_only:
        df = df[~df["Treatment"].isin(NON_SIRNA_TREATMENTS)]
    if drop_controls:
        df = df[~df["Treatment"].astype(str).str.startswith("siNTC")]

    if aggregate:
        # one independent unit per (plate, treatment); mean of features, y is
        # already constant within the group (it's the shared Seahorse value).
        agg = {f: "mean" for f in feat}
        agg[col] = "mean"
        grouped = df.groupby(["Plate", "Treatment"], as_index=False).agg(agg)
        grouped["ImgID"] = (grouped["Plate"].astype(str) + "|"
                            + grouped["Treatment"].astype(str))
        df = grouped

    channel = Path(path).stem.replace("train_", "")
    return VassayData(
        X=df[feat].to_numpy(dtype=np.float64),
        y=df[col].to_numpy(dtype=np.float64),
        plate=df["Plate"].astype(str).to_numpy(),
        treatment=df["Treatment"].astype(str).to_numpy(),
        img_id=df["ImgID"].astype(str).to_numpy(),
        feature_names=feat,
        target=col,
        channel=channel,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else float("nan")
    spear = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "pearson": pear, "spearman": spear, "mae": mae}


# --------------------------------------------------------------------------- #
# Cross-validation schemes (the core of the leakage fix)
# --------------------------------------------------------------------------- #
def cv_splits(
    data: VassayData, scheme: str, n_splits: int = 5, seed: int = 42
):
    """Yield (train_idx, test_idx) for a named CV scheme.

    * ``random``     — shuffled KFold (the *leaky* legacy setting; reproduces the
      optimistic numbers).
    * ``group_plate``— GroupKFold by plate (no plate shared train/test → tests
      generalization to unseen plates / batch).
    * ``group_treatment`` — GroupKFold by treatment (tests generalization to
      unseen perturbations — the real deployment question).
    * ``logo_treatment`` — LeaveOneGroupOut by treatment.
    """
    from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

    if scheme == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from kf.split(data.X)
    elif scheme == "group_plate":
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(data.plate))))
        yield from gkf.split(data.X, data.y, groups=data.plate)
    elif scheme == "group_treatment":
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(data.treatment))))
        yield from gkf.split(data.X, data.y, groups=data.treatment)
    elif scheme == "logo_treatment":
        yield from LeaveOneGroupOut().split(data.X, data.y, groups=data.treatment)
    else:
        raise ValueError(f"unknown cv scheme: {scheme!r}")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_model(name: str, seed: int = 42) -> Callable[[], Any]:
    """Return a *factory* that builds a fresh estimator each fold."""
    name = name.lower()
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return lambda: Ridge(alpha=10.0, random_state=seed)
    if name == "lasso":
        from sklearn.linear_model import Lasso
        return lambda: Lasso(alpha=0.1, max_iter=100000, random_state=seed)
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        return lambda: TabPFNRegressor(device="cpu")
    raise ValueError(f"unknown model: {name!r}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    channel: str
    target: str
    model: str
    scheme: str
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    oof_true: np.ndarray | None = None
    oof_pred: np.ndarray | None = None


def run_cv(
    data: VassayData, model_name: str, scheme: str,
    n_splits: int = 5, seed: int = 42, standardize: bool = True,
    pooled: bool = True,
) -> CVResult:
    """Out-of-fold cross-validation under a given grouping scheme.

    With ``pooled=True`` the headline metrics are computed on the *pooled*
    out-of-fold predictions (all test folds concatenated, scored once). This is
    the correct estimator for small-sample / leave-one-group-out schemes where
    individual folds have too few points to score on their own. ``fold_metrics``
    still holds per-fold scores (for folds with >2 points) for variance info.
    """
    from sklearn.preprocessing import StandardScaler

    factory = make_model(model_name, seed)
    oof_pred = np.full(data.n, np.nan)
    fold_metrics = []
    for tr, te in cv_splits(data, scheme, n_splits, seed):
        Xtr, Xte = data.X[tr], data.X[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        est = factory()
        est.fit(Xtr, data.y[tr])
        pred = est.predict(Xte)
        oof_pred[te] = pred
        if len(te) > 2:
            fold_metrics.append(regression_metrics(data.y[te], pred))

    keys = ("r2", "pearson", "spearman", "mae")
    valid = ~np.isnan(oof_pred)
    if pooled:
        mean = regression_metrics(data.y[valid], oof_pred[valid])
    else:
        mean = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
    std = ({k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys}
           if fold_metrics else {k: float("nan") for k in keys})
    return CVResult(
        channel=data.channel, target=data.target, model=model_name,
        scheme=scheme, metrics_mean=mean, metrics_std=std,
        fold_metrics=fold_metrics, oof_true=data.y, oof_pred=oof_pred,
    )


def baseline_metrics(data: VassayData, scheme: str, seed: int = 42) -> dict[str, float]:
    """Mean-prediction baseline under the same CV scheme (honest floor)."""
    oof = np.full(data.n, np.nan)
    for tr, te in cv_splits(data, scheme, seed=seed):
        oof[te] = data.y[tr].mean()
    return regression_metrics(data.y, oof)
