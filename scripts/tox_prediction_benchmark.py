#!/usr/bin/env python
"""Benchmark tox vs non-tox prediction from DINO, phenotype and pathway features."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dino_pathway_prediction import load_score_matrix  # noqa: E402
from dino_pathway_similarity import html_table, load_signatures_and_dino, rel, short_term  # noqa: E402


PHENO_COLS = ["permito", "mitomass", "area", "intensity", "phenotype_strength", "pred_AUC", "pred_MB"]
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
    parser.add_argument("--out", default="output/2026-06-30/tox_prediction_benchmark")
    parser.add_argument("--top-reactome-terms", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--clip", type=float, default=10.0)
    return parser.parse_args()


def safe_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [col for col in cols if col in df.columns]
    frame = df[keep].apply(pd.to_numeric, errors="coerce")
    return frame.fillna(frame.median(numeric_only=True)).fillna(0.0)


def build_target_meta(target_table: Path) -> pd.DataFrame:
    table = pd.read_csv(target_table)
    target = table[table["category"].eq("Target")].copy().reset_index(drop=True)
    target["tox_rate"] = target["tox_rate"].fillna(0.0)
    target["strict_tox"] = target["tox_rate"].ge(0.3).astype(int)
    target["soft_tox"] = target["tox_rate"].gt(0.0).astype(int)
    target["toxic_collapse_label"] = target["state_class"].eq("toxic_collapse").astype(int)
    return target


def build_dino_feature_sets(args: argparse.Namespace, target: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _, _, _, dino, _ = load_signatures_and_dino(args)
    target_groups = target["group"].astype(str).tolist()
    feature_sets = {}
    dino_frames = {}
    for name, (meta, X) in dino.items():
        dmeta = meta[meta["category"].eq("Target")].reset_index(drop=True)
        positions = meta.index[meta["category"].eq("Target")].to_numpy()
        matrix = X[positions]
        frame = pd.DataFrame(matrix, index=dmeta["group"].astype(str), columns=[f"{name.replace(' ', '_')}_f{i:03d}" for i in range(matrix.shape[1])])
        frame = frame.reindex(target_groups).fillna(0.0)
        dino_frames[name] = frame
        feature_sets[f"DINO {name}"] = frame
    feature_sets["DINO C1+C24"] = pd.concat([dino_frames["C1 brightfield"], dino_frames["C24 mitochondrial"]], axis=1)
    return feature_sets


def build_database_feature_sets(args: argparse.Namespace, target: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pathway_dir = Path(args.pathway_dir)
    target_groups = target["group"].astype(str).tolist()
    feature_sets = {}
    hallmark = load_score_matrix(pathway_dir, "MSigDB_Hallmark_2020")
    reactome = load_score_matrix(pathway_dir, "Reactome_2022", top_n=args.top_reactome_terms)
    feature_sets["DB Hallmark"] = hallmark.reindex(target_groups).fillna(0.0)
    feature_sets["DB Reactome topvar"] = reactome.reindex(target_groups).fillna(0.0)
    feature_sets["DB Hallmark+Reactome"] = pd.concat(
        [feature_sets["DB Hallmark"].add_prefix("hallmark::"), feature_sets["DB Reactome topvar"].add_prefix("reactome::")],
        axis=1,
    )
    return feature_sets


def build_feature_sets(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    target = build_target_meta(Path(args.target_table))
    target_groups = target["group"].astype(str).tolist()
    feature_sets: dict[str, pd.DataFrame] = {}
    feature_sets["Phenotype axes"] = safe_feature_frame(target, PHENO_COLS).set_index(pd.Index(target_groups))
    feature_sets["Curated pathway"] = safe_feature_frame(target, CURATED_PATH_COLS).set_index(pd.Index(target_groups))
    feature_sets["Reference connectivity"] = safe_feature_frame(target, CONNECTIVITY_COLS).set_index(pd.Index(target_groups))
    feature_sets["Curated pathway+connectivity"] = pd.concat(
        [feature_sets["Curated pathway"], feature_sets["Reference connectivity"]], axis=1
    )
    feature_sets.update(build_database_feature_sets(args, target))
    feature_sets.update(build_dino_feature_sets(args, target))
    feature_sets["DINO C24+Phenotype"] = pd.concat([feature_sets["DINO C24 mitochondrial"], feature_sets["Phenotype axes"]], axis=1)
    feature_sets["DINO C24+DB Hallmark"] = pd.concat([feature_sets["DINO C24 mitochondrial"], feature_sets["DB Hallmark"]], axis=1)
    feature_sets["All non-leak features"] = pd.concat(
        [
            feature_sets["DINO C1+C24"],
            feature_sets["DB Hallmark"],
            feature_sets["Curated pathway"],
            feature_sets["Reference connectivity"],
        ],
        axis=1,
    )
    return target, feature_sets


def make_splits(y: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    splits = min(n_splits, positives, negatives)
    if splits < 2:
        raise ValueError("Need at least two positives and two negatives for OOF benchmark")
    return list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=17).split(np.zeros(len(y)), y))


def predict_oof(X: pd.DataFrame, y: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    prob = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    splits = make_splits(y, n_splits)
    matrix = X.to_numpy(dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=0.2,
                class_weight="balanced",
                solver="liblinear",
                random_state=fold,
                max_iter=2000,
            ),
        )
        model.fit(matrix[train_idx], y[train_idx])
        prob[test_idx] = model.predict_proba(matrix[test_idx])[:, 1]
        fold_ids[test_idx] = fold
    return prob, fold_ids


def threshold_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    order = np.argsort(prob)[::-1]
    k = int(y.sum())
    pred_topk = np.zeros_like(y)
    pred_topk[order[:k]] = 1
    pred_05 = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_topk, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "topk_recall": recall_score(y, pred_topk, zero_division=0),
        "topk_precision": precision_score(y, pred_topk, zero_division=0),
        "topk_f1": f1_score(y, pred_topk, zero_division=0),
        "topk_balanced_accuracy": balanced_accuracy_score(y, pred_topk),
        "topk_specificity": specificity,
        "prob05_recall": recall_score(y, pred_05, zero_division=0),
        "prob05_precision": precision_score(y, pred_05, zero_division=0),
    }


def evaluate_feature_set(target: pd.DataFrame, X: pd.DataFrame, label_col: str, n_splits: int) -> tuple[dict[str, float], pd.DataFrame]:
    y = target[label_col].to_numpy(dtype=int)
    prob, fold_ids = predict_oof(X, y, n_splits)
    metrics = {
        "label": label_col,
        "n_targets": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_features": X.shape[1],
        "roc_auc": roc_auc_score(y, prob),
        "average_precision": average_precision_score(y, prob),
    }
    metrics.update(threshold_metrics(y, prob))
    pred = target[["group", "state_class", "tox_rate", "kd_tier", label_col]].copy()
    pred["oof_probability"] = prob
    pred["fold"] = fold_ids
    return metrics, pred


def run_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    target, feature_sets = build_feature_sets(args)
    labels = ["strict_tox", "soft_tox"]
    summary_rows = []
    prediction_frames = []
    for label_col in labels:
        for feature_name, X in feature_sets.items():
            metrics, pred = evaluate_feature_set(target, X, label_col, args.n_splits)
            metrics["feature_set"] = feature_name
            summary_rows.append(metrics)
            pred["label"] = label_col
            pred["feature_set"] = feature_name
            prediction_frames.append(pred)
    summary = pd.DataFrame(summary_rows)
    cols = ["label", "feature_set", "n_targets", "n_positive", "n_negative", "n_features", "roc_auc", "average_precision"]
    extra = [col for col in summary.columns if col not in cols]
    summary = summary[cols + extra].sort_values(["label", "average_precision", "roc_auc"], ascending=[True, False, False])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return {"summary": summary, "predictions": predictions, "target": target}


def plot_metric_bars(summary: pd.DataFrame, figs: Path) -> Path:
    labels = list(summary["label"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(11.8, 5.2 * len(labels)), squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        data = summary[summary["label"].eq(label)].sort_values("average_precision", ascending=False)
        colors = np.where(data["feature_set"].str.contains("DINO"), "#f58518", np.where(data["feature_set"].str.contains("Phenotype"), "#54a24b", "#4c78a8"))
        ax.barh(np.arange(len(data)), data["average_precision"], color=colors, alpha=0.85, label="AP")
        ax.scatter(data["roc_auc"], np.arange(len(data)), color="#1f2937", s=24, label="ROC-AUC")
        ax.set_yticks(np.arange(len(data)))
        ax.set_yticklabels(data["feature_set"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("score")
        ax.set_title(f"{label}: tox vs non-tox prediction")
        ax.grid(axis="x", alpha=0.14)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = figs / "tox_prediction_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_probability_strip(predictions: pd.DataFrame, summary: pd.DataFrame, figs: Path) -> Path:
    best = summary.sort_values(["label", "average_precision", "roc_auc"], ascending=[True, False, False]).groupby("label").head(3)
    keep = predictions.merge(best[["label", "feature_set"]], on=["label", "feature_set"], how="inner")
    panels = list(keep.groupby(["label", "feature_set"]).groups)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.4, 2.3 * len(panels)), squeeze=False)
    for ax, (label, feature_set) in zip(axes.ravel(), panels):
        sub = keep[(keep["label"].eq(label)) & (keep["feature_set"].eq(feature_set))].copy()
        sub = sub.sort_values("oof_probability", ascending=False).reset_index(drop=True)
        y = sub[label].to_numpy(dtype=int)
        colors = np.where(y == 1, "#b2182b", "#9ca3af")
        ax.scatter(np.arange(len(sub)), sub["oof_probability"], c=colors, s=28, alpha=0.85)
        for _, row in sub.head(12).iterrows():
            ax.text(row.name, row["oof_probability"] + 0.025, str(row["group"]), rotation=70, fontsize=7, ha="left", va="bottom")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("OOF p(tox)")
        ax.set_title(f"{label} / {feature_set}")
        ax.grid(axis="y", alpha=0.14)
    fig.tight_layout()
    path = figs / "tox_prediction_probability_strips.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion_heatmap(summary: pd.DataFrame, figs: Path) -> Path:
    data = summary.copy()
    mat = data.pivot_table(index="feature_set", columns="label", values="topk_recall", aggfunc="max")
    mat = mat.loc[summary.groupby("feature_set")["average_precision"].max().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.4, 8.0))
    image = ax.imshow(mat.to_numpy(float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Top-k tox recall, where k = number of positives")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    path = figs / "tox_prediction_topk_recall_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_figures(outputs: dict[str, pd.DataFrame], out: Path) -> dict[str, Path]:
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return {
        "metric_bars": plot_metric_bars(outputs["summary"], figs),
        "probability": plot_probability_strip(outputs["predictions"], outputs["summary"], figs),
        "topk_recall": plot_confusion_heatmap(outputs["summary"], figs),
    }


def render_html(outputs: dict[str, pd.DataFrame], figures: dict[str, Path], out: Path) -> Path:
    summary = outputs["summary"].copy()
    target = outputs["target"]
    best_strict = summary[summary["label"].eq("strict_tox")].sort_values("average_precision", ascending=False).iloc[0]
    best_soft = summary[summary["label"].eq("soft_tox")].sort_values("average_precision", ascending=False).iloc[0]
    tox_targets = target[target["strict_tox"].eq(1)][["group", "state_class", "tox_rate", "kd_tier", "permito", "mitomass", "area", "intensity"]].sort_values("tox_rate", ascending=False)
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tox vs Non-tox Prediction Benchmark</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64748b; --line:#d8dee8; --bg:#f5f7f2; --panel:#fff; --accent:#0f766e; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font-family:Avenir Next, Noto Sans, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:36px 48px 28px; background:linear-gradient(135deg,#f8fafc 0%,#e8efe7 58%,#f7ead8 100%); border-bottom:1px solid var(--line); }}
    main {{ max-width:1180px; margin:0 auto; padding:26px 24px 54px; }} h1 {{ margin:0 0 10px; font-size:32px; letter-spacing:0; }} h2 {{ margin:0 0 14px; font-size:22px; }} p {{ line-height:1.62; }}
    .lead {{ max-width:960px; color:#334155; }} section {{ min-width:0; background:var(--panel); border:1px solid var(--line); margin:18px 0; padding:22px; box-shadow:0 10px 28px rgba(31,41,55,.05); }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-top:18px; }} .metric {{ min-width:0; background:rgba(255,255,255,.72); border:1px solid var(--line); padding:14px 16px; }} .metric b {{ display:block; font-size:24px; }} .metric span {{ color:var(--muted); font-size:13px; }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; align-items:start; }} .two-col > * {{ min-width:0; }}
    figure {{ margin:0; }} figure img {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }} figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
    .callout {{ border-left:4px solid var(--accent); background:#ecfdf5; padding:13px 15px; color:#134e4a; }}
    .table-scroll {{ max-width:100%; min-width:0; overflow-x:auto; border:1px solid var(--line); }} table.data {{ border-collapse:collapse; width:max-content; min-width:100%; max-width:none; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid #e5e7eb; padding:7px 9px; text-align:right; white-space:nowrap; vertical-align:top; }} table.data th:first-child, table.data td:first-child {{ text-align:left; }} table.data td:nth-child(2), table.data td:last-child {{ max-width:420px; white-space:normal; overflow-wrap:anywhere; }} table.data th {{ background:#f8fafc; color:#334155; }} code {{ background:#f1f5f9; padding:2px 5px; }}
    @media (max-width:860px) {{ header {{ padding:26px 22px; }} main {{ padding:18px 14px 42px; }} .grid,.two-col {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>Tox vs Non-tox Prediction Benchmark</h1>
  <p class="lead">比较 DINO、phenotype、curated/database pathway 和 connectivity 特征对 target-level toxicity 的 out-of-fold 分类能力。</p>
  <div class="grid">
    <div class="metric"><b>{len(target)}</b><span>targets</span></div>
    <div class="metric"><b>{int(target['strict_tox'].sum())}</b><span>strict tox positives</span></div>
    <div class="metric"><b>{int(target['soft_tox'].sum())}</b><span>soft tox positives</span></div>
    <div class="metric"><b>{best_strict['average_precision']:.3f}</b><span>best strict AP</span></div>
  </div>
</header>
<main>
  <section>
    <h2>核心结论</h2>
    <div class="callout">strict tox 是小样本任务（6/175），结果主要看 PR-AUC/AP 和 top-k recall。最佳 strict tox 特征是 {html.escape(str(best_strict['feature_set']))}: AP={best_strict['average_precision']:.3f}, ROC-AUC={best_strict['roc_auc']:.3f}。最佳 soft tox 特征是 {html.escape(str(best_soft['feature_set']))}: AP={best_soft['average_precision']:.3f}, ROC-AUC={best_soft['roc_auc']:.3f}。</div>
    <p>Phenotype 轴接近上限，因为 tox label 本身来自 imaging cell-count loss；DINO-only 和 pathway-only 更能反映是否能提前从表征/机制层识别 toxicity-like perturbation。</p>
  </section>

  <section class="two-col">
    <figure><img src="{rel(figures['metric_bars'], out)}" alt="tox prediction metrics"><figcaption>柱为 average precision，黑点为 ROC-AUC。</figcaption></figure>
    <figure><img src="{rel(figures['topk_recall'], out)}" alt="top-k recall"><figcaption>按预测概率取 top-k（k=阳性数）时能召回多少 tox target。</figcaption></figure>
  </section>

  <section>
    <h2>OOF Probability Strips</h2>
    <figure><img src="{rel(figures['probability'], out)}" alt="OOF probability strips"><figcaption>每个 panel 是一个 label 下表现最好的 3 个 feature set；红点是真阳性。</figcaption></figure>
  </section>

  <section class="two-col">
    <div><h2>Benchmark Summary</h2><div class="table-scroll">{html_table(summary, max_rows=40)}</div></div>
    <div><h2>Strict Tox Targets</h2><div class="table-scroll">{html_table(tox_targets, max_rows=12)}</div></div>
  </section>

  <section>
    <h2>Files</h2>
    <p>Outputs: <code>tox_prediction_summary.csv</code>, <code>tox_prediction_oof_predictions.csv</code>, and figures under <code>figs/</code>.</p>
    <p>Labels: <code>strict_tox = tox_rate >= 0.3</code>; <code>soft_tox = tox_rate > 0</code>. Models are class-balanced L2 logistic regression with stratified out-of-fold prediction.</p>
  </section>
</main>
</body>
</html>
"""
    path = out / "tox_prediction_benchmark_report.html"
    path.write_text(text, encoding="utf-8")
    return path


def save_outputs(outputs: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    outputs["summary"].to_csv(out / "tox_prediction_summary.csv", index=False)
    outputs["predictions"].to_csv(out / "tox_prediction_oof_predictions.csv", index=False)
    outputs["target"].to_csv(out / "tox_prediction_target_labels.csv", index=False)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    outputs = run_benchmark(args)
    save_outputs(outputs, out)
    figures = make_figures(outputs, out)
    report = render_html(outputs, figures, out)
    print(f"[tox_prediction_benchmark] wrote {report}")
    print(outputs["summary"].sort_values(["label", "average_precision"], ascending=[True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())