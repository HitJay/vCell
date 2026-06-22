#!/usr/bin/env python
"""Phase 2 for the B+A route: leakage-aware multimodal baseline ladder.

Benchmarks simple, defensible baselines before any vCell extension:

* expression-only (within-NTC z HVG)
* brightfield C1 DINOv2
* mitochondrial C24 DINOv2
* expression + C1
* expression + C1 + C24

Each feature set is evaluated with grouped out-of-fold prediction. Headline
metrics are computed at the target level because EE hit calling is a target
ranking problem; well-level metrics are retained as diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


Y_COLS = {
    "permito": "pheno_permito_dpsi_z",
    "mitomass": "pheno_mitomass_z",
    "area": "pheno_area_z",
}


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    keys: tuple[str, ...]


FEATURE_BLOCKS = (
    FeatureBlock("expression", ("X_zscore_hvg",)),
    FeatureBlock("c1_brightfield", ("X_dino_c1",)),
    FeatureBlock("c24_mito", ("X_dino_c24",)),
    FeatureBlock("expression_c1", ("X_zscore_hvg", "X_dino_c1")),
    FeatureBlock("expression_c1_c24", ("X_zscore_hvg", "X_dino_c1", "X_dino_c24")),
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"r2": np.nan, "pearson": np.nan, "spearman": np.nan, "mae": np.nan}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    pearson = pearsonr(y_true, y_pred)[0] if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else np.nan
    spearman = spearmanr(y_true, y_pred)[0] if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "pearson": pearson, "spearman": spearman, "mae": mae}


def topk_enrichment(y_true: np.ndarray, y_pred: np.ndarray, k: int, hit_abs: float = 5.0) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"topk_precision": np.nan, "topk_recall": np.nan, "n_true_hits": 0}
    k = min(k, len(y_true))
    true_hits = np.abs(y_true) >= hit_abs
    order = np.argsort(-np.abs(y_pred))[:k]
    n_true = int(true_hits.sum())
    found = int(true_hits[order].sum())
    recall = found / n_true if n_true else np.nan
    return {"topk_precision": found / k, "topk_recall": recall, "n_true_hits": n_true}


def get_matrix(adata, keys: tuple[str, ...]) -> np.ndarray:
    mats = []
    for key in keys:
        if key in adata.obsm:
            arr = adata.obsm[key]
        elif key in adata.layers:
            arr = adata.layers[key]
        else:
            raise KeyError(f"feature block {key!r} not found in AnnData")
        if hasattr(arr, "todense"):
            arr = np.asarray(arr.todense())
        else:
            arr = np.asarray(arr)
        mats.append(arr.astype(np.float32, copy=False))
    return np.concatenate(mats, axis=1) if len(mats) > 1 else mats[0]


def cv_indices(obs: pd.DataFrame, scheme: str, seed: int, n_splits: int):
    from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

    idx = np.arange(len(obs))
    if scheme == "random":
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx)
    elif scheme == "group_plate":
        groups = obs["plate"].astype(str).to_numpy()
        yield from GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(idx, groups=groups)
    elif scheme == "logo_target":
        groups = obs["group"].astype(str).to_numpy()
        yield from LeaveOneGroupOut().split(idx, groups=groups)
    else:
        raise ValueError(f"unknown scheme: {scheme}")


def fit_predict_oof(X: np.ndarray, y: np.ndarray, obs: pd.DataFrame, scheme: str, seed: int, n_splits: int):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.full_like(y, np.nan, dtype=np.float64)
    base = np.full_like(y, np.nan, dtype=np.float64)
    for train_idx, test_idx in cv_indices(obs, scheme, seed, n_splits):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0, solver="lsqr", random_state=seed),
        )
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
        base[test_idx] = np.nanmean(y[train_idx], axis=0)
    return pred, base


def target_level_frame(obs: pd.DataFrame, y: np.ndarray, pred: np.ndarray, feature_set: str, scheme: str, model: str) -> pd.DataFrame:
    rows = []
    base = obs.copy()
    base["_row"] = np.arange(len(base))
    for yi, outcome in enumerate(Y_COLS):
        tmp = base.assign(y_true=y[:, yi], y_pred=pred[:, yi])
        tmp = tmp[(tmp["category"].astype(str) == "Target") & tmp["y_pred"].notna()]
        agg = tmp.groupby("group", as_index=False).agg(
            category=("category", "first"),
            n_wells=("group", "size"),
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            tox_rate=("tox_flag", "mean"),
        )
        agg["outcome"] = outcome
        agg["feature_set"] = feature_set
        agg["scheme"] = scheme
        agg["model"] = model
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def score_predictions(obs: pd.DataFrame, y: np.ndarray, pred: np.ndarray, feature_set: str, scheme: str, model: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    target_df = target_level_frame(obs, y, pred, feature_set, scheme, model)
    target_mask = obs["category"].astype(str).eq("Target").to_numpy()
    for yi, outcome in enumerate(Y_COLS):
        well_metrics = regression_metrics(y[target_mask, yi], pred[target_mask, yi])
        tsub = target_df[target_df["outcome"] == outcome]
        target_metrics = regression_metrics(tsub["y_true"].to_numpy(), tsub["y_pred"].to_numpy())
        top10 = topk_enrichment(tsub["y_true"].to_numpy(), tsub["y_pred"].to_numpy(), k=10)
        top20 = topk_enrichment(tsub["y_true"].to_numpy(), tsub["y_pred"].to_numpy(), k=20)
        row = {
            "feature_set": feature_set,
            "scheme": scheme,
            "model": model,
            "outcome": outcome,
            "n_wells": int(target_mask.sum()),
            "n_targets": int(len(tsub)),
        }
        row.update({f"well_{k}": v for k, v in well_metrics.items()})
        row.update({f"target_{k}": v for k, v in target_metrics.items()})
        row.update({f"top10_{k}": v for k, v in top10.items()})
        row.update({f"top20_{k}": v for k, v in top20.items()})
        rows.append(row)
    return rows, target_df


def write_plot(metrics: pd.DataFrame, out_dir: Path) -> None:
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    model_df = metrics[metrics["model"] == "ridge"].copy()
    outcomes = list(Y_COLS.keys())
    schemes = [s for s in ["group_plate", "logo_target"] if s in set(model_df["scheme"])]
    fig, axes = plt.subplots(len(outcomes), len(schemes), figsize=(5.2 * len(schemes), 3.6 * len(outcomes)), sharey=True)
    axes = np.asarray(axes).reshape(len(outcomes), len(schemes))
    for i, outcome in enumerate(outcomes):
        for j, scheme in enumerate(schemes):
            ax = axes[i, j]
            sub = model_df[(model_df["outcome"] == outcome) & (model_df["scheme"] == scheme)]
            sub = sub.sort_values("target_spearman")
            ax.barh(sub["feature_set"], sub["target_spearman"], color="#4c78a8")
            ax.axvline(0, color="grey", lw=0.7)
            ax.set_title(f"{outcome} / {scheme}")
            ax.set_xlabel("target-level Spearman")
    fig.suptitle("B+A Phase 2 baseline ladder (target-level OOF ranking)")
    fig.tight_layout()
    fig.savefig(figs / "multimodal_baseline_benchmark.png", dpi=150)
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    main = metrics[(metrics["model"] == "ridge") & (metrics["scheme"] == "logo_target")].copy()
    main = main.sort_values(["outcome", "target_spearman"], ascending=[True, False])
    cols = [
        "outcome",
        "feature_set",
        "target_spearman",
        "target_pearson",
        "target_mae",
        "top10_topk_precision",
        "top20_topk_precision",
    ]
    md = f"""# B+A Phase 2 - Multimodal Baseline Benchmark

Generated from `data/processed/adata_multimodal.h5ad`.

## Headline: leave-target-out target-level ranking

{main[cols].round(3).to_markdown(index=False)}

## Notes

- `logo_target` is the main anti-leakage test for new-target generalization.
- `group_plate` is retained as the plate-generalization diagnostic.
- `mean_baseline` rows in the CSV are fold-wise training-mean predictors under the same split.
- C24 is an upper-bound feature block because it directly observes mitochondrial dyes related to the phenotype.
"""
    (out_dir / "B_A_multimodal_benchmark_summary.md").write_text(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    parser.add_argument("--schemes", nargs="+", default=["group_plate", "logo_target"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args(argv)

    import anndata as ad

    adata = ad.read_h5ad(args.adata)
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["plate"] = obs["plate"].astype(str)
    obs["category"] = obs["category"].astype(str)
    obs["qc_fail"] = obs["qc_fail"].astype(bool) if "qc_fail" in obs else False
    obs["tox_flag"] = obs["tox_flag"].astype(bool) if "tox_flag" in obs else False
    keep = (~obs["qc_fail"]) & (~obs["tox_flag"]) & obs["category"].isin(["Target", "PC", "NC"])
    adata = adata[keep.to_numpy()].copy()
    obs = obs.loc[keep].reset_index(drop=True)
    y = np.column_stack([obs[col].to_numpy(dtype=float) for col in Y_COLS.values()])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for block in FEATURE_BLOCKS:
        X = get_matrix(adata, block.keys)
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        Xf = X[finite]
        yf = y[finite]
        obsf = obs.loc[finite].reset_index(drop=True)
        for scheme in args.schemes:
            pred, base = fit_predict_oof(Xf, yf, obsf, scheme, args.seed, args.n_splits)
            rows, tdf = score_predictions(obsf, yf, pred, block.name, scheme, "ridge")
            metrics_rows.extend(rows)
            prediction_frames.append(tdf)
            base_rows, base_tdf = score_predictions(obsf, yf, base, block.name, scheme, "mean_baseline")
            metrics_rows.extend(base_rows)
            prediction_frames.append(base_tdf)
            print(f"[ba_benchmark] {block.name:18s} {scheme:12s} n={len(obsf)}")

    metrics = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(out_dir / "multimodal_baseline_benchmark.csv", index=False)
    pred_df.to_csv(out_dir / "multimodal_oof_target_predictions.csv", index=False)
    write_plot(metrics, out_dir)
    write_summary(metrics, out_dir)
    print(f"[ba_benchmark] wrote {out_dir / 'multimodal_baseline_benchmark.csv'}")
    print(f"[ba_benchmark] wrote {out_dir / 'multimodal_oof_target_predictions.csv'}")
    print(f"[ba_benchmark] wrote {out_dir / 'B_A_multimodal_benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())