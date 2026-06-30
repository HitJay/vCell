#!/usr/bin/env python
"""Predict TMRM/MitoTracker phenotype axes from DRUG-seq transcript features."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dino_pathway_prediction import load_score_matrix  # noqa: E402
from dino_pathway_similarity import html_table, rel, short_term  # noqa: E402


OUTCOMES = {
    "permito": "per-mito dPsi",
    "mitomass": "MitoTracker mass",
    "area": "TMRM area",
    "intensity": "TMRM intensity",
}
CURATED_PATH_COLS = [
    "path_OXPHOS_ETC",
    "path_MITO_BIOGENESIS",
    "path_FAO_LIPID",
    "path_AMPK_MTOR_INSULIN",
    "path_ISR_ER_STRESS",
    "path_PROTEOSTASIS_AUTOPHAGY",
    "path_APOPTOSIS_TOXICITY",
]
CONNECTIVITY_COLS = ["conn_BAM15", "conn_MK8722", "conn_PSMC3", "conn_ATP5B", "conn_SLC25A4", "conn_toxicity_margin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--target-table", default="output/2026-06-29/pathway_phenotype_correlation/crossmodal_moa_target_table.csv")
    parser.add_argument("--pathway-dir", default="output/2026-06-30/dino_pathway_similarity")
    parser.add_argument("--out", default="output/2026-06-30/transcript_tmrm_prediction")
    parser.add_argument("--top-reactome-terms", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.nanstd(a[ok]) < 1e-10 or np.nanstd(b[ok]) < 1e-10:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0] if method == "spearman" else pearsonr(a[ok], b[ok])[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 4:
        return {"spearman": np.nan, "pearson": np.nan, "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "spearman": corr(y_true, y_pred, "spearman"),
        "pearson": corr(y_true, y_pred, "pearson"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def topk_hit_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int, hit_abs: float = 5.0) -> dict[str, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"precision": np.nan, "recall": np.nan, "n_true_hits": 0}
    true_hit = np.abs(y_true) >= hit_abs
    k = min(k, len(y_true))
    selected = np.argsort(-np.abs(y_pred))[:k]
    found = int(true_hit[selected].sum())
    n_true = int(true_hit.sum())
    return {"precision": found / k, "recall": found / n_true if n_true else np.nan, "n_true_hits": n_true}


def target_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    target = table[table["category"].eq("Target")].copy().reset_index(drop=True)
    for outcome in OUTCOMES:
        target[outcome] = pd.to_numeric(target[outcome], errors="coerce")
    return target


def load_target_obsm_features(adata_path: Path, target: pd.DataFrame) -> dict[str, pd.DataFrame]:
    adata = ad.read_h5ad(adata_path)
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["category"] = obs["category"].astype(str)
    obs["qc_fail"] = as_bool(obs["qc_fail"])
    obs["tox_flag"] = as_bool(obs["tox_flag"])
    target_groups = target["group"].astype(str).tolist()
    feature_sets = {}
    specs = [
        ("Transcript HVG clean", "X_zscore_hvg", (~obs["qc_fail"]) & (~obs["tox_flag"]) & obs["category"].eq("Target")),
        ("Transcript HVG QC-all", "X_zscore_hvg", (~obs["qc_fail"]) & obs["category"].eq("Target")),
        ("DINO C1 comparator", "X_dino_c1", (~obs["qc_fail"]) & obs["category"].eq("Target")),
        ("DINO C24 comparator", "X_dino_c24", (~obs["qc_fail"]) & obs["category"].eq("Target")),
    ]
    for name, key, mask in specs:
        matrix = np.asarray(adata.obsm[key], dtype=np.float32)
        obs_sub = obs.loc[mask].reset_index(drop=True)
        mat_sub = matrix[mask.to_numpy()]
        rows = []
        vectors = []
        for group, idx in obs_sub.groupby("group").indices.items():
            rows.append(str(group))
            vectors.append(mat_sub[np.asarray(idx)].mean(axis=0))
        frame = pd.DataFrame(np.vstack(vectors), index=rows, columns=[f"{safe_name(name)}_f{i:04d}" for i in range(matrix.shape[1])])
        feature_sets[name] = frame.reindex(target_groups).fillna(0.0)
    feature_sets["DINO C1+C24 comparator"] = pd.concat(
        [feature_sets["DINO C1 comparator"], feature_sets["DINO C24 comparator"]], axis=1
    )
    return feature_sets


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def safe_frame(target: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    groups = target["group"].astype(str).tolist()
    keep = [col for col in cols if col in target.columns]
    frame = target[keep].apply(pd.to_numeric, errors="coerce")
    return frame.fillna(frame.median(numeric_only=True)).fillna(0.0).set_index(pd.Index(groups))


def load_database_features(pathway_dir: Path, target: pd.DataFrame, top_reactome_terms: int) -> dict[str, pd.DataFrame]:
    groups = target["group"].astype(str).tolist()
    hallmark = load_score_matrix(pathway_dir, "MSigDB_Hallmark_2020").reindex(groups).fillna(0.0)
    reactome = load_score_matrix(pathway_dir, "Reactome_2022", top_n=top_reactome_terms).reindex(groups).fillna(0.0)
    return {
        "Transcript DB Hallmark": hallmark,
        "Transcript DB Reactome topvar": reactome,
        "Transcript DB Hallmark+Reactome": pd.concat([hallmark.add_prefix("hallmark::"), reactome.add_prefix("reactome::")], axis=1),
    }


def build_feature_sets(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    target = target_table(Path(args.target_table))
    feature_sets = load_target_obsm_features(Path(args.adata), target)
    feature_sets["Transcript curated pathway"] = safe_frame(target, CURATED_PATH_COLS)
    feature_sets["Transcript connectivity"] = safe_frame(target, CONNECTIVITY_COLS)
    feature_sets["Transcript curated+connectivity"] = pd.concat(
        [feature_sets["Transcript curated pathway"], feature_sets["Transcript connectivity"]], axis=1
    )
    feature_sets.update(load_database_features(Path(args.pathway_dir), target, args.top_reactome_terms))
    feature_sets["Transcript HVG clean+DB Hallmark"] = pd.concat(
        [feature_sets["Transcript HVG clean"], feature_sets["Transcript DB Hallmark"]], axis=1
    )
    return target, feature_sets


def make_splits(target: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = target["state_class"].fillna("unknown").astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    splits = max(2, min(n_splits, int(counts.min())))
    return list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=23).split(np.zeros(len(target)), labels))


def fit_predict_oof(X: pd.DataFrame, Y: np.ndarray, target: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full_like(Y, np.nan, dtype=np.float32)
    fold_ids = np.full(Y.shape[0], -1, dtype=int)
    matrix = X.to_numpy(dtype=float)
    for fold, (train_idx, test_idx) in enumerate(make_splits(target, args.n_splits), start=1):
        model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        model.fit(matrix[train_idx], Y[train_idx])
        pred[test_idx] = model.predict(matrix[test_idx]).astype(np.float32)
        fold_ids[test_idx] = fold
    return pred, fold_ids


def mean_baseline(Y: np.ndarray, target: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full_like(Y, np.nan, dtype=np.float32)
    fold_ids = np.full(Y.shape[0], -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(make_splits(target, args.n_splits), start=1):
        pred[test_idx] = np.nanmean(Y[train_idx], axis=0)
        fold_ids[test_idx] = fold
    return pred, fold_ids


def evaluate(target: pd.DataFrame, Y: np.ndarray, pred: np.ndarray, fold_ids: np.ndarray, feature_set: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for outcome_idx, outcome in enumerate(OUTCOMES):
        y_true = Y[:, outcome_idx]
        y_pred = pred[:, outcome_idx]
        metrics = regression_metrics(y_true, y_pred)
        top10 = topk_hit_metrics(y_true, y_pred, k=10)
        top20 = topk_hit_metrics(y_true, y_pred, k=20)
        row = {
            "feature_set": feature_set,
            "model": model,
            "outcome": outcome,
            "outcome_label": OUTCOMES[outcome],
            "n_targets": len(target),
        }
        row.update(metrics)
        row.update({f"top10_{k}": v for k, v in top10.items()})
        row.update({f"top20_{k}": v for k, v in top20.items()})
        metric_rows.append(row)
        tmp = target[["group", "state_class", "kd_tier", "tox_rate"]].copy()
        tmp["feature_set"] = feature_set
        tmp["model"] = model
        tmp["outcome"] = outcome
        tmp["fold"] = fold_ids
        tmp["y_true"] = y_true
        tmp["y_pred"] = y_pred
        pred_rows.append(tmp)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def profile_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, model, group), sub in predictions.groupby(["feature_set", "model", "group"]):
        sub = sub.set_index("outcome").reindex(list(OUTCOMES))
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "group": group,
                "state_class": sub["state_class"].dropna().iloc[0] if sub["state_class"].notna().any() else np.nan,
                "profile_spearman": corr(y_true, y_pred, "spearman"),
                "profile_pearson": corr(y_true, y_pred, "pearson"),
                "profile_cosine": float(cosine_similarity(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0, 0])
                if np.isfinite(y_true).all() and np.isfinite(y_pred).all()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    target, feature_sets = build_feature_sets(args)
    Y = target[list(OUTCOMES)].to_numpy(dtype=np.float32)
    metrics = []
    predictions = []
    base_pred, base_fold = mean_baseline(Y, target, args)
    base_metrics, base_predictions = evaluate(target, Y, base_pred, base_fold, "Mean baseline", "fold_mean")
    metrics.append(base_metrics)
    predictions.append(base_predictions)
    for feature_set, X in feature_sets.items():
        pred, fold_ids = fit_predict_oof(X, Y, target, args)
        metric_df, pred_df = evaluate(target, Y, pred, fold_ids, feature_set, "ridge")
        metrics.append(metric_df)
        predictions.append(pred_df)
    metric_table = pd.concat(metrics, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    profiles = profile_metrics(prediction_table)
    return {"metrics": metric_table, "predictions": prediction_table, "profiles": profiles, "target": target}


def plot_metric_heatmap(metrics: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    data = metrics[metrics["model"].eq("ridge")].copy()
    mat = data.pivot_table(index="feature_set", columns="outcome_label", values="spearman", aggfunc="mean")
    order = data.groupby("feature_set")["spearman"].median().sort_values(ascending=False).index
    mat = mat.reindex(order)
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    image = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-0.25, vmax=0.65, aspect="auto")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Transcript/DINO features predicting TMRM-related phenotype axes")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="OOF target Spearman")
    fig.tight_layout()
    path = figs / "transcript_tmrm_prediction_spearman_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_bars(metrics: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    data = metrics[metrics["model"].eq("ridge")].copy()
    outcomes = list(OUTCOMES)
    fig, axes = plt.subplots(len(outcomes), 1, figsize=(10.4, 3.7 * len(outcomes)), squeeze=False)
    for ax, outcome in zip(axes.ravel(), outcomes):
        sub = data[data["outcome"].eq(outcome)].sort_values("spearman", ascending=True)
        colors = np.where(sub["feature_set"].str.contains("DINO"), "#f58518", "#4c78a8")
        ax.barh(sub["feature_set"], sub["spearman"], color=colors, alpha=0.85)
        ax.axvline(0, color="#6b7280", lw=0.8)
        ax.set_xlabel("OOF target Spearman")
        ax.set_title(OUTCOMES[outcome])
    fig.tight_layout()
    path = figs / "transcript_tmrm_prediction_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter(predictions: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    top = metrics[metrics["model"].eq("ridge")].sort_values("spearman", ascending=False).groupby("outcome").head(1)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = predictions[
            predictions["feature_set"].eq(row["feature_set"])
            & predictions["model"].eq("ridge")
            & predictions["outcome"].eq(row["outcome"])
        ]
        for state, state_df in sub.groupby("state_class", dropna=False):
            ax.scatter(state_df["y_true"], state_df["y_pred"], s=28, alpha=0.74, label=str(state))
        lo = np.nanmin([sub["y_true"].min(), sub["y_pred"].min()])
        hi = np.nanmax([sub["y_true"].max(), sub["y_pred"].max()])
        ax.plot([lo, hi], [lo, hi], color="#6b7280", lw=0.8)
        ax.set_title(f"{OUTCOMES[row['outcome']]}\n{row['feature_set']} / rho={row['spearman']:.2f}", fontsize=9)
        ax.set_xlabel("actual")
        ax.set_ylabel("OOF predicted")
        ax.grid(alpha=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = figs / "transcript_tmrm_prediction_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_profile_box(profiles: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    data = profiles[profiles["model"].eq("ridge")].copy()
    order = data.groupby("feature_set")["profile_spearman"].median().sort_values(ascending=False).index
    values = [data[data["feature_set"].eq(name)]["profile_spearman"].dropna().to_numpy() for name in order]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.boxplot(values, tick_labels=list(order), showfliers=False)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylabel("Per-target 4-axis profile Spearman")
    ax.set_title("Can transcript features reconstruct the TMRM phenotype profile?")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    fig.tight_layout()
    path = figs / "transcript_tmrm_profile_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_figures(outputs: dict[str, pd.DataFrame], out: Path) -> dict[str, Path]:
    (out / "figs").mkdir(parents=True, exist_ok=True)
    return {
        "heatmap": plot_metric_heatmap(outputs["metrics"], out),
        "bars": plot_top_bars(outputs["metrics"], out),
        "scatter": plot_scatter(outputs["predictions"], outputs["metrics"], out),
        "profile": plot_profile_box(outputs["profiles"], out),
    }


def render_html(outputs: dict[str, pd.DataFrame], figures: dict[str, Path], out: Path) -> Path:
    metrics = outputs["metrics"].copy()
    ridge = metrics[metrics["model"].eq("ridge")].copy()
    best = ridge.sort_values("spearman", ascending=False).iloc[0]
    transcript_only = ridge[~ridge["feature_set"].str.contains("DINO")].copy()
    best_transcript = transcript_only.sort_values("spearman", ascending=False).iloc[0]
    summary = ridge.sort_values(["outcome", "spearman"], ascending=[True, False])[
        ["outcome_label", "feature_set", "spearman", "pearson", "r2", "mae", "top10_precision", "top20_precision"]
    ]
    profile_summary = outputs["profiles"][outputs["profiles"]["model"].eq("ridge")].groupby("feature_set").agg(
        median_profile_spearman=("profile_spearman", "median"),
        mean_profile_spearman=("profile_spearman", "mean"),
        median_profile_cosine=("profile_cosine", "median"),
    ).reset_index().sort_values("median_profile_spearman", ascending=False)
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DRUG-seq Transcript to TMRM Prediction</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64748b; --line:#d8dee8; --bg:#f5f7f2; --panel:#fff; --accent:#0f766e; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font-family:Avenir Next, Noto Sans, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:36px 48px 28px; background:linear-gradient(135deg,#f8fafc 0%,#e8efe7 58%,#f7ead8 100%); border-bottom:1px solid var(--line); }}
    main {{ max-width:1180px; margin:0 auto; padding:26px 24px 54px; }} h1 {{ margin:0 0 10px; font-size:32px; letter-spacing:0; }} h2 {{ margin:0 0 14px; font-size:22px; }} p {{ line-height:1.62; }}
    .lead {{ max-width:960px; color:#334155; }} section {{ min-width:0; background:var(--panel); border:1px solid var(--line); margin:18px 0; padding:22px; box-shadow:0 10px 28px rgba(31,41,55,.05); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:18px; }} .metric {{ min-width:0; background:rgba(255,255,255,.72); border:1px solid var(--line); padding:14px 16px; }} .metric b {{ display:block; font-size:24px; }} .metric span {{ color:var(--muted); font-size:13px; }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; align-items:start; }} .two-col > * {{ min-width:0; }}
    figure {{ margin:0; }} figure img {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }} figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
    .callout {{ border-left:4px solid var(--accent); background:#ecfdf5; padding:13px 15px; color:#134e4a; }}
    .table-scroll {{ max-width:100%; min-width:0; overflow-x:auto; border:1px solid var(--line); }} table.data {{ border-collapse:collapse; width:max-content; min-width:100%; max-width:none; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid #e5e7eb; padding:7px 9px; text-align:right; white-space:nowrap; vertical-align:top; }} table.data th:first-child, table.data td:first-child {{ text-align:left; }} table.data td:nth-child(2), table.data td:last-child {{ max-width:420px; white-space:normal; overflow-wrap:anywhere; }} table.data th {{ background:#f8fafc; color:#334155; }} code {{ background:#f1f5f9; padding:2px 5px; }}
    @media (max-width:860px) {{ header {{ padding:26px 22px; }} main {{ padding:18px 14px 42px; }} .grid,.two-col {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>DRUG-seq Transcript to TMRM Prediction</h1>
  <p class="lead">用 DRUG-seq transcript signature / pathway activity 预测 target-level TMRM 与 MitoTracker phenotype 轴，并用 DINO 特征作为影像上限参照。</p>
  <div class="grid">
    <div class="metric"><b>{int(best['n_targets'])}</b><span>targets</span></div>
    <div class="metric"><b>{best_transcript['spearman']:.3f}</b><span>best transcript-only Spearman</span></div>
    <div class="metric"><b>{best['spearman']:.3f}</b><span>best overall Spearman</span></div>
  </div>
</header>
<main>
  <section>
    <h2>核心结论</h2>
    <div class="callout">Transcript 能预测一部分 TMRM/MitoTracker phenotype，最强 transcript-only 组合是 {html.escape(str(best_transcript['feature_set']))} → {html.escape(str(best_transcript['outcome_label']))}, Spearman={best_transcript['spearman']:.3f}。DINO 作为影像上限通常更强，最佳整体是 {html.escape(str(best['feature_set']))} → {html.escape(str(best['outcome_label']))}, Spearman={best['spearman']:.3f}。</div>
    <p>这里的 transcript 特征包含 HVG target signature、curated pathway、database pathway 和 reference connectivity。评估是 target-level stratified out-of-fold Ridge，避免同一 target 同时出现在训练和测试。</p>
  </section>

  <section class="two-col">
    <figure><img src="{rel(figures['heatmap'], out)}" alt="spearman heatmap"><figcaption>各特征集预测四个 TMRM/MitoTracker phenotype 轴的 OOF Spearman。</figcaption></figure>
    <figure><img src="{rel(figures['profile'], out)}" alt="profile boxplot"><figcaption>每个 target 的 4-axis phenotype profile 是否能被预测出来。</figcaption></figure>
  </section>

  <section>
    <h2>Outcome-wise Ranking</h2>
    <figure><img src="{rel(figures['bars'], out)}" alt="outcome bars"><figcaption>DINO 橙色为影像参照；蓝色为 transcript-derived 特征。</figcaption></figure>
  </section>

  <section>
    <h2>Best Actual vs Predicted</h2>
    <figure><img src="{rel(figures['scatter'], out)}" alt="actual vs predicted"><figcaption>每个 outcome 下表现最好的 feature set 的 held-out scatter。</figcaption></figure>
  </section>

  <section class="two-col">
    <div><h2>OOF Metrics</h2><div class="table-scroll">{html_table(summary, max_rows=44)}</div></div>
    <div><h2>Profile Metrics</h2><div class="table-scroll">{html_table(profile_summary, max_rows=20)}</div></div>
  </section>

  <section>
    <h2>Files</h2>
    <p>Outputs: <code>transcript_tmrm_prediction_metrics.csv</code>, <code>transcript_tmrm_prediction_oof_predictions.csv</code>, <code>transcript_tmrm_prediction_profile_metrics.csv</code>, and figures under <code>figs/</code>.</p>
  </section>
</main>
</body>
</html>
"""
    path = out / "transcript_tmrm_prediction_report.html"
    path.write_text(text, encoding="utf-8")
    return path


def save_outputs(outputs: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    outputs["metrics"].to_csv(out / "transcript_tmrm_prediction_metrics.csv", index=False)
    outputs["predictions"].to_csv(out / "transcript_tmrm_prediction_oof_predictions.csv", index=False)
    outputs["profiles"].to_csv(out / "transcript_tmrm_prediction_profile_metrics.csv", index=False)
    outputs["target"].to_csv(out / "transcript_tmrm_prediction_target_table.csv", index=False)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    outputs = run_benchmark(args)
    save_outputs(outputs, out)
    figures = make_figures(outputs, out)
    report = render_html(outputs, figures, out)
    print(f"[transcript_tmrm_prediction] wrote {report}")
    main_metrics = outputs["metrics"][outputs["metrics"]["model"].eq("ridge")]
    print(main_metrics.sort_values(["outcome", "spearman"], ascending=[True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())