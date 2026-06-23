#!/usr/bin/env python
"""Build the EE-DrugSeq-v1 benchmark environment and HTML report.

This script creates a first benchmark package from the processed multimodal
AnnData object. It focuses on leakage-resistant baseline scoring and a readable
HTML report, so future virtual-cell models can be compared against transparent
baselines before any model-specific integration.

Run with the scvi/anndata environment:

    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/ee_drugseq_benchmark.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vcell.benchmark import (  # noqa: E402
    expression_benchmark_metrics,
    leave_one_out_mean_baseline,
    nearest_neighbor_delta_baseline,
    zero_delta_baseline,
)

PHENO_COLS = ["permito", "mitomass", "area"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--phenotype-benchmark", default="output/2026-06-22/ba_multimodal_plan/multimodal_baseline_benchmark.csv")
    parser.add_argument("--dino-separation", default="output/2026-06-22/ba_multimodal_plan/dino_state_separation_summary.csv")
    parser.add_argument("--tox-safety", default="output/2026-06-22/ba_multimodal_plan/EE_strict_tox_safety_report_status.md")
    parser.add_argument("--out", default="output/2026-06-22/vcell_benchmark_research")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-target-wells", type=int, default=4)
    return parser.parse_args()


def as_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)


def get_target_expression_delta(adata, table: pd.DataFrame, min_target_wells: int) -> tuple[pd.DataFrame, np.ndarray]:
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["category"] = obs["category"].astype(str)
    if "qc_fail" in obs:
        keep = ~obs["qc_fail"].astype(bool).to_numpy()
    else:
        keep = np.ones(adata.n_obs, dtype=bool)

    X = as_dense(adata.obsm["X_zscore_hvg"]).astype(np.float64)
    X = X[keep]
    obs = obs.loc[keep].reset_index(drop=True)

    table_meta = table[["group", "state_class", "recommendation", "tox_rate", "kd_tier"]].drop_duplicates("group")
    rows = []
    deltas = []
    for group, idx in obs.groupby("group", sort=True).indices.items():
        sub = obs.iloc[np.asarray(idx)]
        category = str(sub["category"].mode().iloc[0])
        if category != "Target" or len(idx) < min_target_wells:
            continue
        deltas.append(X[np.asarray(idx)].mean(axis=0))
        rows.append({
            "group": group,
            "category": category,
            "n_wells_expression": int(len(idx)),
            "tox_rate_well": float(sub["tox_flag"].astype(bool).mean()) if "tox_flag" in sub else np.nan,
        })
    meta = pd.DataFrame(rows).merge(table_meta, on="group", how="left")
    return meta, np.vstack(deltas)


def get_target_feature_matrix(adata, groups: list[str], obsm_key: str) -> np.ndarray:
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    X = as_dense(adata.obsm[obsm_key]).astype(np.float64)
    rows = []
    for group in groups:
        mask = obs["group"].eq(group).to_numpy()
        rows.append(X[mask].mean(axis=0))
    feat = np.vstack(rows)
    scale = feat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (feat - feat.mean(axis=0)) / scale


def get_pheno_feature_matrix(meta: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    pheno = table[["group"] + PHENO_COLS].drop_duplicates("group").set_index("group")
    arr = pheno.reindex(meta["group"])[PHENO_COLS].to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = arr.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (arr - arr.mean(axis=0)) / scale


def run_expression_baselines(adata, table: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, true_delta = get_target_expression_delta(adata, table, args.min_target_wells)
    groups = meta["group"].tolist()
    baselines: dict[str, np.ndarray] = {
        "zero_delta_ntc": zero_delta_baseline(true_delta),
        "loo_mean_delta": leave_one_out_mean_baseline(true_delta),
    }
    for name, obsm_key in [("c1_nearest_delta", "X_dino_c1"), ("c24_nearest_delta", "X_dino_c24")]:
        features = get_target_feature_matrix(adata, groups, obsm_key)
        pred, neighbor = nearest_neighbor_delta_baseline(features, true_delta)
        baselines[name] = pred
        meta[f"{name}_neighbor"] = [groups[int(i)] for i in neighbor]
    pheno_features = get_pheno_feature_matrix(meta, table)
    pred, neighbor = nearest_neighbor_delta_baseline(pheno_features, true_delta)
    baselines["phenotype_nearest_delta"] = pred
    meta["phenotype_nearest_delta_neighbor"] = [groups[int(i)] for i in neighbor]

    score_rows = []
    for model, pred_delta in baselines.items():
        score = expression_benchmark_metrics(true_delta, pred_delta, model=model, top_k=args.top_k).as_dict()
        score["task"] = "expression_delta"
        score_rows.append(score)
    scores = pd.DataFrame(score_rows).sort_values(["pds", "des_topk"], ascending=False)
    return scores, meta


def write_task_spec(out: Path) -> None:
    spec = """# EE-DrugSeq-v1 Benchmark Specification

## Dataset

- Unit: target-level HepG2 siRNA perturbation response derived from matched mini-bulk DRUG-seq, DINOv2 imaging, and TMRM/MitoTracker phenotypes.
- Control: NTC wells.
- Primary split: leave-target-out for new-target generalization.
- Diagnostic split: group-plate for batch generalization.
- Random splits are leakage audits only.

## Tasks

### A. Expression Delta Prediction

Predict target-level expression deltas relative to same-batch NTC. Metrics: mini-bulk top-|delta| overlap (DES-topK), perturbation discrimination score (PDS), MAE, delta Pearson and delta Spearman.

### B. Cross-modal Phenotype Prediction

Predict target-level mitochondrial phenotypes (`permito`, `area`, `mitomass`). Metrics: target Spearman/Pearson, MAE, top-k hit recovery.

### C. MoA State Recovery

Recover `neutral_or_uncertain`, `uncoupler_like`, `mixed_uncoupling_biogenesis`, `biogenesis_like`, and `toxic_collapse`. Metrics: macro-F1, balanced accuracy, and toxic-collapse sensitivity.

### D. Toxicity Triage

Predict toxic-collapse or high tox-rate. Metrics: AUROC/AUPRC where model scores are available, plus anchor-target review against the safety report.

## Baseline Ladder

1. zero_delta_ntc: no perturbation effect.
2. loo_mean_delta: mean of other target deltas.
3. c1_nearest_delta: copy expression delta from nearest target in brightfield DINO space.
4. c24_nearest_delta: copy expression delta from nearest target in mitochondrial DINO space.
5. phenotype_nearest_delta: copy expression delta from nearest target in measured phenotype space.
6. Ridge/PLS multimodal baselines from the existing B+A benchmark.
"""
    (out / "benchmark_spec.md").write_text(spec, encoding="utf-8")
    tasks_yaml = """name: EE-DrugSeq-v1
primary_split: logo_target
diagnostic_split: group_plate
tasks:
  expression_delta:
    metrics: [des_topk, pds, mae, delta_pearson_mean, delta_spearman_mean]
  phenotype_prediction:
    outcomes: [permito, area, mitomass]
    metrics: [target_spearman, target_pearson, target_mae, topk_precision]
  moa_state_recovery:
    labels: [neutral_or_uncertain, uncoupler_like, mixed_uncoupling_biogenesis, biogenesis_like, toxic_collapse]
    metrics: [macro_f1, balanced_accuracy]
  toxicity_triage:
    labels: [toxic_collapse]
    metrics: [auroc, auprc, anchor_target_review]
"""
    (out / "benchmark_tasks.yaml").write_text(tasks_yaml, encoding="utf-8")


def plot_expression_scores(scores: pd.DataFrame, out: Path) -> None:
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    plot_df = scores.sort_values("pds")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].barh(plot_df["model"], plot_df["pds"], color="#1f77b4")
    axes[0].set_title("PDS")
    axes[1].barh(plot_df["model"], plot_df["des_topk"], color="#2ca02c")
    axes[1].set_title("DES top-k overlap")
    axes[2].barh(plot_df["model"], plot_df["mae"], color="#d62728")
    axes[2].set_title("MAE (lower is better)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("EE-DrugSeq-v1 expression-delta baseline scores")
    fig.tight_layout()
    fig.savefig(figs / "expression_delta_baseline_scores.png", dpi=160)
    plt.close(fig)


def load_optional_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def html_table(df: pd.DataFrame, *, max_rows: int = 20, float_format: str = "{:.3f}") -> str:
    if df.empty:
        return "<p class='muted'>Not available.</p>"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else float_format.format(float(x)))
    return view.to_html(index=False, escape=True, classes="data")


def render_html(
    out: Path,
    expression_scores: pd.DataFrame,
    target_meta: pd.DataFrame,
    phenotype: pd.DataFrame,
    dino_sep: pd.DataFrame,
    tox_status_md: str,
) -> None:
    phenotype_main = pd.DataFrame()
    if not phenotype.empty:
        phenotype_main = phenotype[(phenotype["model"].eq("ridge")) & (phenotype["scheme"].eq("logo_target"))].copy()
        keep = ["outcome", "feature_set", "target_spearman", "target_pearson", "target_mae", "top10_topk_precision"]
        phenotype_main = phenotype_main[[c for c in keep if c in phenotype_main.columns]].sort_values(
            ["outcome", "target_spearman"], ascending=[True, False]
        )

    tox_html = "<p class='muted'>Not available.</p>"
    if tox_status_md:
        lines = [html.escape(line) for line in tox_status_md.splitlines() if line.strip()]
        tox_html = "<pre>" + "\n".join(lines[:42]) + "</pre>"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EE-DrugSeq-v1 Benchmark Report</title>
  <style>
    :root {{ --ink:#17212b; --muted:#5e6a75; --line:#d7dee6; --paper:#f6f8fa; --blue:#0b4f7a; --green:#2f7a4f; --red:#a33d3d; }}
    body {{ margin:0; font-family:"Aptos","Segoe UI",sans-serif; color:var(--ink); background:var(--paper); line-height:1.5; }}
    main {{ width:min(1180px, calc(100vw - 36px)); margin:24px auto 48px; }}
    section, header {{ background:white; border:1px solid var(--line); border-radius:8px; padding:22px; margin:0 0 16px; box-shadow:0 8px 20px rgba(23,33,43,.05); }}
    h1 {{ margin:0 0 10px; font-size:34px; }} h2 {{ margin:0 0 12px; color:var(--blue); }} h3 {{ margin:18px 0 8px; }}
    .lead {{ font-size:18px; max-width:88ch; color:#263747; }} .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:16px; }}
    .metric {{ border:1px solid var(--line); border-radius:7px; padding:12px; background:#fbfdff; }}
    .metric strong {{ display:block; font-size:26px; color:var(--blue); }} .metric span {{ color:var(--muted); font-size:13px; }}
    table.data {{ border-collapse:collapse; width:100%; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }} table.data th {{ background:#eef4f8; color:var(--blue); }}
    figure {{ margin:12px 0; }} img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; background:white; }}
    pre {{ white-space:pre-wrap; background:#f3f6f8; border:1px solid var(--line); border-radius:7px; padding:12px; font-size:12px; }}
    a {{ color:var(--blue); font-weight:700; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="muted">Generated 2026-06-22 from <code>data/processed/adata_multimodal.h5ad</code></p>
    <h1>EE-DrugSeq-v1 Benchmark Report</h1>
    <p class="lead">A phenotype-grounded validation environment for virtual-cell and perturbation-response models. The benchmark asks whether models predict held-out HepG2 siRNA expression responses, preserve perturbation identity, and explain downstream mitochondrial phenotypes and toxic-collapse biology.</p>
    <div class="grid">
      <div class="metric"><strong>{int(target_meta['group'].nunique())}</strong><span>scored target perturbations</span></div>
      <div class="metric"><strong>{int(expression_scores['n_genes'].max())}</strong><span>expression HVG features</span></div>
      <div class="metric"><strong>{len(expression_scores)}</strong><span>expression baselines</span></div>
      <div class="metric"><strong>logo_target</strong><span>primary anti-leakage split</span></div>
    </div>
  </header>

  <section>
    <h2>Benchmark Plan</h2>
    <p>The benchmark has four tasks: expression-delta prediction, mitochondrial phenotype prediction, MoA-state recovery, and toxicity triage. The first runnable version scores expression-delta baselines and imports the existing B+A phenotype benchmark as the phenotype leaderboard.</p>
    <ul>
      <li><strong>Primary validation:</strong> leave-target-out / held-out target generalization.</li>
      <li><strong>Diagnostic validation:</strong> group-plate generalization for batch sensitivity.</li>
      <li><strong>Baseline ladder:</strong> zero-delta NTC, leave-one-out mean delta, nearest-neighbor deltas from C1, C24 and phenotype spaces, plus existing ridge multimodal baselines.</li>
    </ul>
  </section>

  <section>
    <h2>Expression Delta Baseline Scores</h2>
    <figure><img src="figs/expression_delta_baseline_scores.png" alt="Expression delta baseline scores"></figure>
    {html_table(expression_scores[['model','des_topk','pds','mae','delta_pearson_mean','delta_spearman_mean','n_targets','n_genes']], max_rows=20)}
  </section>

  <section>
    <h2>Phenotype Prediction Baseline</h2>
    <p class="muted">Imported from the existing B+A multimodal benchmark. C24 is treated as an upper-bound image feature because it directly observes TMRM/MitoTracker dyes.</p>
    {html_table(phenotype_main, max_rows=20)}
  </section>

  <section>
    <h2>DINO Geometry and Toxicity</h2>
    <p>DINO geometry is supporting evidence, not the primary benchmark score. C24 t-SNE/UMAP helps visualize the toxic-collapse local neighborhood; high-dimensional centroid distances remain the quantitative backing.</p>
    <figure><img src="../ba_multimodal_plan/figs/dino_c24_tsne_state_map.png" alt="C24 DINO t-SNE state map"></figure>
    {html_table(dino_sep, max_rows=10)}
  </section>

  <section>
    <h2>Toxicity Safety Cross-check</h2>
    {tox_html}
  </section>

  <section>
    <h2>Artifacts</h2>
    <ul>
      <li><a href="benchmark_spec.md">benchmark_spec.md</a></li>
    <li><a href="benchmark_environment.md">benchmark_environment.md</a></li>
      <li><a href="benchmark_tasks.yaml">benchmark_tasks.yaml</a></li>
      <li><a href="expression_delta_benchmark_scores.csv">expression_delta_benchmark_scores.csv</a></li>
      <li><a href="expression_delta_target_metadata.csv">expression_delta_target_metadata.csv</a></li>
      <li><a href="vcell_drugseq_benchmark_research.md">research positioning note</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    (out / "EE_DrugSeq_v1_benchmark_report.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    import anndata as ad

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.table)
    adata = ad.read_h5ad(args.adata)

    expression_scores, target_meta = run_expression_baselines(adata, table, args)
    expression_scores.to_csv(out / "expression_delta_benchmark_scores.csv", index=False)
    target_meta.to_csv(out / "expression_delta_target_metadata.csv", index=False)
    write_task_spec(out)
    plot_expression_scores(expression_scores, out)

    phenotype = load_optional_table(args.phenotype_benchmark)
    dino_sep = load_optional_table(args.dino_separation)
    tox_status = Path(args.tox_safety).read_text(encoding="utf-8") if Path(args.tox_safety).exists() else ""
    render_html(out, expression_scores, target_meta, phenotype, dino_sep, tox_status)
    manifest = {
        "benchmark": "EE-DrugSeq-v1",
        "adata": args.adata,
        "target_table": args.table,
        "n_expression_targets": int(target_meta["group"].nunique()),
        "n_expression_genes": int(expression_scores["n_genes"].max()),
        "outputs": [
            "benchmark_spec.md",
            "benchmark_tasks.yaml",
            "expression_delta_benchmark_scores.csv",
            "expression_delta_target_metadata.csv",
            "EE_DrugSeq_v1_benchmark_report.html",
        ],
    }
    (out / "benchmark_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ee_benchmark] wrote {out / 'EE_DrugSeq_v1_benchmark_report.html'}")
    print(f"[ee_benchmark] wrote {out / 'expression_delta_benchmark_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())