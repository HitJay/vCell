#!/usr/bin/env python
"""Predict database pathway activity profiles from DINOv2 features."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dino_pathway_similarity import html_table, load_signatures_and_dino, rel, short_term  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--target-table", default="output/2026-06-29/pathway_phenotype_correlation/crossmodal_moa_target_table.csv")
    parser.add_argument("--pathway-dir", default="output/2026-06-30/dino_pathway_similarity")
    parser.add_argument("--out", default="output/2026-06-30/dino_pathway_prediction")
    parser.add_argument("--top-reactome-terms", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--clip", type=float, default=10.0)
    return parser.parse_args()


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full_like(pvalues, np.nan)
    ok = np.isfinite(pvalues)
    if ok.sum() == 0:
        return qvalues
    p = pvalues[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    qvalues[ok] = restored
    return qvalues


def corr_pair(actual: np.ndarray, pred: np.ndarray, method: str) -> tuple[float, float]:
    ok = np.isfinite(actual) & np.isfinite(pred)
    if ok.sum() < 4 or np.nanstd(actual[ok]) < 1e-10 or np.nanstd(pred[ok]) < 1e-10:
        return np.nan, np.nan
    if method == "spearman":
        rho, pvalue = spearmanr(actual[ok], pred[ok])
    else:
        rho, pvalue = pearsonr(actual[ok], pred[ok])
    return float(rho), float(pvalue)


def load_score_matrix(pathway_dir: Path, library: str, top_n: int | None = None) -> pd.DataFrame:
    path = pathway_dir / f"database_gene_set_scores_{safe_name(library)}.csv"
    long = pd.read_csv(path)
    target = long[long["category"].eq("Target")].copy()
    matrix = target.pivot_table(index="group", columns="term", values="mean_signature_z", aggfunc="mean")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    if top_n is not None and matrix.shape[1] > top_n:
        keep = matrix.var(axis=0).sort_values(ascending=False).head(top_n).index
        matrix = matrix.loc[:, keep]
    return matrix


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def build_pathway_matrices(pathway_dir: Path, top_reactome_terms: int) -> dict[str, pd.DataFrame]:
    hallmark = load_score_matrix(pathway_dir, "MSigDB_Hallmark_2020")
    reactome = load_score_matrix(pathway_dir, "Reactome_2022", top_n=top_reactome_terms)
    combined = pd.concat(
        [
            hallmark.add_prefix("MSigDB_Hallmark_2020: "),
            reactome.add_prefix("Reactome_2022: "),
        ],
        axis=1,
    )
    return {
        "MSigDB_Hallmark_2020": hallmark,
        "Reactome_2022_topvar": reactome,
        "Combined_database_selected": combined,
    }


def build_dino_feature_sets(args: argparse.Namespace) -> tuple[dict[str, tuple[pd.DataFrame, np.ndarray]], pd.DataFrame]:
    sig_meta, _, _, dino, _ = load_signatures_and_dino(args)
    features: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for name, (meta, X) in dino.items():
        target = meta[meta["category"].eq("Target")].reset_index(drop=True)
        positions = meta.index[meta["category"].eq("Target")].to_numpy()
        features[name] = (target, X[positions])
    c1_meta, c1 = features["C1 brightfield"]
    c24_meta, c24 = features["C24 mitochondrial"]
    common = sorted(set(c1_meta["group"]) & set(c24_meta["group"]))
    c1_pos = c1_meta.reset_index().set_index("group").loc[common, "index"].to_numpy()
    c24_pos = c24_meta.reset_index().set_index("group").loc[common, "index"].to_numpy()
    combined_meta = c1_meta.iloc[c1_pos].reset_index(drop=True)
    features["C1+C24 combined"] = (combined_meta, np.hstack([c1[c1_pos], c24[c24_pos]]))
    return features, sig_meta


def align_X_y(feature_meta: pd.DataFrame, X: np.ndarray, y_matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    common = [group for group in feature_meta["group"].astype(str) if group in y_matrix.index]
    positions = feature_meta.reset_index().set_index("group").loc[common, "index"].to_numpy()
    meta = feature_meta.iloc[positions].reset_index(drop=True)
    Y = y_matrix.loc[common].to_numpy(dtype=np.float32)
    terms = y_matrix.columns.astype(str).tolist()
    return meta, X[positions], Y, terms


def make_splits(meta: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = meta["state_class"].fillna("unknown").astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    max_splits = int(min(n_splits, counts.min())) if not counts.empty else n_splits
    max_splits = max(2, max_splits)
    splitter = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=11)
    return list(splitter.split(np.zeros(len(labels)), labels))


def oof_predict(
    X: np.ndarray,
    Y: np.ndarray,
    meta: pd.DataFrame,
    model_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[int]]:
    pred = np.full_like(Y, np.nan, dtype=np.float32)
    fold_ids = np.full(Y.shape[0], -1, dtype=int)
    splits = make_splits(meta, args.n_splits)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X[train_idx])
        X_test = x_scaler.transform(X[test_idx])
        Y_train = y_scaler.fit_transform(Y[train_idx])
        if model_name == "ridge":
            model = Ridge(alpha=args.ridge_alpha)
        elif model_name == "knn_distance":
            model = KNeighborsRegressor(
                n_neighbors=min(args.neighbor_k, len(train_idx)),
                weights="distance",
                metric="euclidean",
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        model.fit(X_train, Y_train)
        pred[test_idx] = y_scaler.inverse_transform(model.predict(X_test)).astype(np.float32)
        fold_ids[test_idx] = fold
    return pred, fold_ids.tolist()


def evaluate_prediction(
    meta: pd.DataFrame,
    Y: np.ndarray,
    pred: np.ndarray,
    terms: list[str],
    library: str,
    feature_set: str,
    model: str,
    fold_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term_rows = []
    for i, term in enumerate(terms):
        actual = Y[:, i]
        predicted = pred[:, i]
        sp, sp_p = corr_pair(actual, predicted, "spearman")
        pr, pr_p = corr_pair(actual, predicted, "pearson")
        err = predicted - actual
        term_rows.append(
            {
                "library": library,
                "feature_set": feature_set,
                "model": model,
                "term": term,
                "n_targets": len(meta),
                "spearman": sp,
                "spearman_p": sp_p,
                "pearson": pr,
                "pearson_p": pr_p,
                "mae": float(np.nanmean(np.abs(err))),
                "rmse": float(np.sqrt(np.nanmean(err**2))),
                "actual_sd": float(np.nanstd(actual)),
            }
        )
    term_metrics = pd.DataFrame(term_rows)
    term_metrics["spearman_fdr"] = fdr_bh(term_metrics["spearman_p"].to_numpy())

    profile_rows = []
    for row_idx, row in meta.iterrows():
        sp, _ = corr_pair(Y[row_idx], pred[row_idx], "spearman")
        pr, _ = corr_pair(Y[row_idx], pred[row_idx], "pearson")
        cosine = cosine_similarity(Y[row_idx], pred[row_idx])
        profile_rows.append(
            {
                "library": library,
                "feature_set": feature_set,
                "model": model,
                "group": row["group"],
                "state_class": row.get("state_class", np.nan),
                "kd_tier": row.get("kd_tier", np.nan),
                "fold": fold_ids[row_idx],
                "profile_spearman": sp,
                "profile_pearson": pr,
                "profile_cosine": cosine,
            }
        )
    profile_metrics = pd.DataFrame(profile_rows)

    pred_rows = []
    top_terms = term_metrics.sort_values("spearman", ascending=False).head(30)["term"].tolist()
    term_to_idx = {term: i for i, term in enumerate(terms)}
    for row_idx, row in meta.iterrows():
        for term in top_terms:
            i = term_to_idx[term]
            pred_rows.append(
                {
                    "library": library,
                    "feature_set": feature_set,
                    "model": model,
                    "group": row["group"],
                    "state_class": row.get("state_class", np.nan),
                    "term": term,
                    "actual": float(Y[row_idx, i]),
                    "predicted": float(pred[row_idx, i]),
                    "fold": fold_ids[row_idx],
                }
            )
    predictions = pd.DataFrame(pred_rows)
    return term_metrics, profile_metrics, predictions


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    av = a[ok]
    bv = b[ok]
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-12:
        return np.nan
    return float(np.dot(av, bv) / denom)


def run_prediction(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    pathway_matrices = build_pathway_matrices(Path(args.pathway_dir), args.top_reactome_terms)
    features, _ = build_dino_feature_sets(args)
    term_tables = []
    profile_tables = []
    prediction_tables = []
    for library, y_matrix in pathway_matrices.items():
        for feature_set, (feature_meta, X) in features.items():
            meta, aligned_X, Y, terms = align_X_y(feature_meta, X, y_matrix)
            for model in ["ridge", "knn_distance"]:
                pred, fold_ids = oof_predict(aligned_X, Y, meta, model, args)
                term_metrics, profile_metrics, predictions = evaluate_prediction(
                    meta, Y, pred, terms, library, feature_set, model, fold_ids
                )
                term_tables.append(term_metrics)
                profile_tables.append(profile_metrics)
                prediction_tables.append(predictions)
    term = pd.concat(term_tables, ignore_index=True)
    profile = pd.concat(profile_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = summarize(term, profile)
    return {"summary": summary, "term_metrics": term, "profile_metrics": profile, "predictions": predictions}


def summarize(term: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    term_summary = term.groupby(["library", "feature_set", "model"]).agg(
        n_terms=("term", "count"),
        median_term_spearman=("spearman", "median"),
        mean_term_spearman=("spearman", "mean"),
        frac_terms_spearman_gt_0_2=("spearman", lambda x: float(np.nanmean(x > 0.2))),
        frac_terms_fdr_lt_0_05=("spearman_fdr", lambda x: float(np.nanmean(x < 0.05))),
        median_term_pearson=("pearson", "median"),
        median_rmse=("rmse", "median"),
    ).reset_index()
    profile_summary = profile.groupby(["library", "feature_set", "model"]).agg(
        median_profile_spearman=("profile_spearman", "median"),
        mean_profile_spearman=("profile_spearman", "mean"),
        median_profile_cosine=("profile_cosine", "median"),
    ).reset_index()
    return term_summary.merge(profile_summary, on=["library", "feature_set", "model"], how="left")


def save_outputs(outputs: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    outputs["summary"].to_csv(out / "dino_pathway_prediction_summary.csv", index=False)
    outputs["term_metrics"].to_csv(out / "dino_pathway_prediction_term_metrics.csv", index=False)
    outputs["profile_metrics"].to_csv(out / "dino_pathway_prediction_profile_metrics.csv", index=False)
    outputs["predictions"].to_csv(out / "dino_pathway_prediction_topterm_oof_predictions.csv", index=False)


def plot_summary(summary: pd.DataFrame, figs: Path) -> Path:
    data = summary.copy()
    data["label"] = data["library"].map(short_term) + "\n" + data["feature_set"] + "\n" + data["model"]
    data = data.sort_values("median_term_spearman", ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    colors = data["feature_set"].map({"C1 brightfield": "#4c78a8", "C24 mitochondrial": "#f58518", "C1+C24 combined": "#54a24b"}).fillna("#777777")
    ax.bar(np.arange(len(data)), data["median_term_spearman"], color=colors, alpha=0.85)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Median held-out term Spearman")
    ax.set_title("Can DINO features predict database pathway activity?")
    fig.tight_layout()
    path = figs / "dino_pathway_prediction_summary_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_terms(term: pd.DataFrame, figs: Path) -> Path:
    best_terms = term.groupby("term")["spearman"].max().sort_values(ascending=False).head(28).index
    data = term[term["term"].isin(best_terms)].copy()
    data["row"] = data["term"].map(short_term)
    data["col"] = data["library"].map(short_term) + " / " + data["feature_set"] + " / " + data["model"]
    mat = data.pivot_table(index="row", columns="col", values="spearman", aggfunc="max")
    mat = mat.reindex(index=[short_term(term) for term in best_terms])
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    image = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-0.25, vmax=0.65, aspect="auto")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("Most DINO-predictable pathway activities")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    path = figs / "dino_pathway_prediction_top_terms_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_profile_box(profile: pd.DataFrame, figs: Path) -> Path:
    data = profile.copy()
    keys = list(data.groupby(["library", "feature_set", "model"]).groups)
    values = [data[(data["library"].eq(lib)) & (data["feature_set"].eq(fs)) & (data["model"].eq(model))]["profile_spearman"].dropna().to_numpy() for lib, fs, model in keys]
    labels = [f"{short_term(lib)}\n{fs}\n{model}" for lib, fs, model in keys]
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylabel("Held-out target profile Spearman")
    ax.set_title("Per-target pathway profile prediction from DINO features")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout()
    path = figs / "dino_pathway_prediction_profile_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_scatter(predictions: pd.DataFrame, term_metrics: pd.DataFrame, figs: Path) -> Path:
    best = term_metrics.sort_values("spearman", ascending=False).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, best.iterrows()):
        sub = predictions[
            predictions["library"].eq(row["library"])
            & predictions["feature_set"].eq(row["feature_set"])
            & predictions["model"].eq(row["model"])
            & predictions["term"].eq(row["term"])
        ]
        for state, ssub in sub.groupby("state_class", dropna=False):
            ax.scatter(ssub["actual"], ssub["predicted"], s=28, alpha=0.72, label=str(state))
        lo = float(np.nanmin([sub["actual"].min(), sub["predicted"].min()]))
        hi = float(np.nanmax([sub["actual"].max(), sub["predicted"].max()]))
        ax.plot([lo, hi], [lo, hi], color="#6b7280", lw=0.8)
        ax.set_title(f"{short_term(row['term'])}\n{row['feature_set']} / {row['model']} / rho={row['spearman']:.2f}", fontsize=8)
        ax.set_xlabel("actual")
        ax.set_ylabel("OOF predicted")
        ax.grid(alpha=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = figs / "dino_pathway_prediction_top_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_figures(outputs: dict[str, pd.DataFrame], out: Path) -> dict[str, Path]:
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return {
        "summary": plot_summary(outputs["summary"], figs),
        "top_terms": plot_top_terms(outputs["term_metrics"], figs),
        "profile": plot_profile_box(outputs["profile_metrics"], figs),
        "scatter": plot_top_scatter(outputs["predictions"], outputs["term_metrics"], figs),
    }


def render_html(outputs: dict[str, pd.DataFrame], figures: dict[str, Path], out: Path) -> Path:
    summary = outputs["summary"].sort_values("median_term_spearman", ascending=False)
    term = outputs["term_metrics"].sort_values("spearman", ascending=False)
    profile = outputs["profile_metrics"]
    best = summary.iloc[0]
    best_profile = profile[
        profile["library"].eq(best["library"])
        & profile["feature_set"].eq(best["feature_set"])
        & profile["model"].eq(best["model"])
    ]
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DINOv2 Pathway Prediction Benchmark</title>
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
    .table-scroll {{ max-width:100%; min-width:0; overflow-x:auto; border:1px solid var(--line); }} table.data {{ border-collapse:collapse; width:max-content; min-width:100%; max-width:none; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid #e5e7eb; padding:7px 9px; text-align:right; white-space:nowrap; vertical-align:top; }} table.data th:first-child, table.data td:first-child {{ text-align:left; }} table.data td:nth-child(4), table.data td:last-child {{ max-width:460px; white-space:normal; overflow-wrap:anywhere; }} table.data th {{ background:#f8fafc; color:#334155; }} code {{ background:#f1f5f9; padding:2px 5px; }}
    @media (max-width:860px) {{ header {{ padding:26px 22px; }} main {{ padding:18px 14px 42px; }} .grid,.two-col {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>DINOv2 Pathway Prediction Benchmark</h1>
  <p class="lead">用 DINOv2 C1/C24 target-level features 预测数据库 pathway activity，评估是否可以从影像 embedding 直接恢复 perturbation 的 pathway-like transcriptional profile。</p>
  <div class="grid">
    <div class="metric"><b>{int(best['n_terms'])}</b><span>terms in best library profile</span></div>
    <div class="metric"><b>{best['median_term_spearman']:.3f}</b><span>best median term Spearman</span></div>
    <div class="metric"><b>{best_profile['profile_spearman'].median():.3f}</b><span>best median target profile Spearman</span></div>
  </div>
</header>
<main>
  <section>
    <h2>核心结论</h2>
    <div class="callout">可以预测，但更像“部分 pathway/state readout 可预测”，不是所有 pathway 都可从 DINO 直接恢复。当前最强组合是 {html.escape(str(best['library']))} / {html.escape(str(best['feature_set']))} / {html.escape(str(best['model']))}，median term Spearman={best['median_term_spearman']:.3f}。</div>
    <p>Ridge 代表 supervised mapping，KNN-distance 代表“空间近邻找类似 pathway perturbation”。如果 KNN 表现接近 Ridge，说明 DINO 几何本身已经带有 pathway neighborhood 信息；如果 Ridge 明显更强，说明存在可学习但不一定局部成团的映射。</p>
  </section>

  <section class="two-col">
    <figure><img src="{rel(figures['summary'], out)}" alt="prediction summary"><figcaption>各 DINO feature set / model / pathway library 的 held-out term prediction 表现。</figcaption></figure>
    <figure><img src="{rel(figures['profile'], out)}" alt="profile prediction"><figcaption>每个 held-out target 的 pathway profile 预测相关性。</figcaption></figure>
  </section>

  <section>
    <h2>最可预测的 pathway terms</h2>
    <figure><img src="{rel(figures['top_terms'], out)}" alt="top predictable terms"><figcaption>按最佳 Spearman 排序的 pathway terms；颜色越红代表 DINO 越能预测该 term activity。</figcaption></figure>
  </section>

  <section>
    <h2>Top Term Actual vs Predicted</h2>
    <figure><img src="{rel(figures['scatter'], out)}" alt="actual vs predicted"><figcaption>前 6 个最佳 term 的 held-out actual vs predicted scatter。</figcaption></figure>
  </section>

  <section class="two-col">
    <div><h2>Summary Metrics</h2><div class="table-scroll">{html_table(summary, max_rows=30)}</div></div>
    <div><h2>Top Terms</h2><div class="table-scroll">{html_table(term[['library','feature_set','model','term','spearman','spearman_fdr','pearson','rmse']], max_rows=24)}</div></div>
  </section>

  <section>
    <h2>Files</h2>
    <p>Outputs: <code>dino_pathway_prediction_summary.csv</code>, <code>dino_pathway_prediction_term_metrics.csv</code>, <code>dino_pathway_prediction_profile_metrics.csv</code>, <code>dino_pathway_prediction_topterm_oof_predictions.csv</code>.</p>
    <p>Model: target-level stratified 5-fold out-of-fold prediction. Inputs are batch NTC-z DINO centroids. Outputs are database gene-set activity scores from the previous DINO-pathway analysis.</p>
  </section>
</main>
</body>
</html>
"""
    path = out / "dino_pathway_prediction_report.html"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    outputs = run_prediction(args)
    save_outputs(outputs, out)
    figures = make_figures(outputs, out)
    report = render_html(outputs, figures, out)
    print(f"[dino_pathway_prediction] wrote {report}")
    print(outputs["summary"].sort_values("median_term_spearman", ascending=False).head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())