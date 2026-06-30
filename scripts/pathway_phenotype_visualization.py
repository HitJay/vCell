#!/usr/bin/env python
"""Render an HTML visualization for pathway-phenotype correlations."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


PATHWAY_COLS = [
    "path_OXPHOS_ETC",
    "path_MITO_BIOGENESIS",
    "path_FAO_LIPID",
    "path_AMPK_MTOR_INSULIN",
    "path_ISR_ER_STRESS",
    "path_PROTEOSTASIS_AUTOPHAGY",
    "path_APOPTOSIS_TOXICITY",
]
PHENO_COLS = ["permito", "mitomass", "area", "intensity"]

PATHWAY_LABELS = {
    "path_OXPHOS_ETC": "OXPHOS/ETC",
    "path_MITO_BIOGENESIS": "Mito biogenesis",
    "path_FAO_LIPID": "FAO/lipid",
    "path_AMPK_MTOR_INSULIN": "AMPK/mTOR/insulin",
    "path_ISR_ER_STRESS": "ISR/ER stress",
    "path_PROTEOSTASIS_AUTOPHAGY": "Proteostasis/autophagy",
    "path_APOPTOSIS_TOXICITY": "Apoptosis/toxicity",
}
PHENO_LABELS = {
    "permito": "per-mito dPsi",
    "mitomass": "MitoTracker mass",
    "area": "TMRM area",
    "intensity": "TMRM intensity",
}
STATE_COLORS = {
    "uncoupler_like": "#d95f02",
    "mixed_uncoupling_biogenesis": "#7570b3",
    "biogenesis_like": "#1b9e77",
    "energizer_like": "#66a61e",
    "toxic_collapse": "#b2182b",
    "neutral_or_uncertain": "#8c8c8c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/2026-06-29/pathway_phenotype_correlation")
    parser.add_argument("--target-table", default=None)
    parser.add_argument("--correlations-long", default=None)
    parser.add_argument("--spearman-matrix", default=None)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def html_table(df: pd.DataFrame, *, max_rows: int = 20, float_format: str = "{:.3f}") -> str:
    view = df.head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else float_format.format(value))
    return view.to_html(index=False, escape=True, classes="data")


def relpath(path: Path, out: Path) -> str:
    return html.escape(path.relative_to(out).as_posix())


def numeric_frame(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frame = table[cols].apply(pd.to_numeric, errors="coerce")
    return frame.fillna(frame.median(numeric_only=True)).fillna(0.0)


def compute_tsne(table: pd.DataFrame, feature_cols: list[str], random_state: int) -> pd.DataFrame:
    usable_cols = [column for column in feature_cols if column in table]
    features = numeric_frame(table, usable_cols)
    scaled = StandardScaler().fit_transform(features)
    perplexity = min(30, max(5, (len(table) - 1) // 4))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
    ).fit_transform(scaled)
    coords = table[
        [
            "group",
            "category",
            "state_class",
            "kd_tier",
            "tox_rate",
            "consensus_score",
            "phenotype_strength",
            "dominant_axis",
            "recommendation",
        ]
    ].copy()
    coords["tsne_1"] = embedding[:, 0]
    coords["tsne_2"] = embedding[:, 1]
    coords["tsne_feature_set"] = "pathway_plus_phenotype"
    coords["tsne_perplexity"] = perplexity
    return coords


def plot_tsne(coords: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    for state, subset in coords.groupby("state_class", dropna=False):
        color = STATE_COLORS.get(str(state), "#6b7280")
        sizes = 36 + 12 * subset["phenotype_strength"].fillna(0).clip(0, 10)
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=sizes,
            c=color,
            alpha=0.74,
            edgecolor="white",
            linewidth=0.6,
            label=str(state),
        )
    label_pool = coords.sort_values(["consensus_score", "phenotype_strength"], ascending=False).head(18)
    for _, row in label_pool.iterrows():
        ax.annotate(
            str(row["group"]),
            (row["tsne_1"], row["tsne_2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="#d1d5db", lw=0.7)
    ax.axvline(0, color="#d1d5db", lw=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Target t-SNE from pathway + phenotype features")
    ax.legend(frameon=False, fontsize=7, loc="best", markerscale=0.8)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    path = figs / "pathway_phenotype_tsne_state_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_pair_scatter(table: pd.DataFrame, correlations: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    top = correlations.sort_values("abs_spearman_r", ascending=False).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    axes = axes.ravel()
    for axis_index, (_, row) in enumerate(top.iterrows()):
        ax = axes[axis_index]
        pathway_col = f"path_{row['pathway']}"
        pheno_col = str(row["phenotype"])
        for state, subset in table.groupby("state_class", dropna=False):
            color = STATE_COLORS.get(str(state), "#6b7280")
            ax.scatter(
                subset[pathway_col],
                subset[pheno_col],
                s=24,
                alpha=0.68,
                c=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axhline(0, color="#d1d5db", lw=0.6)
        ax.axvline(0, color="#d1d5db", lw=0.6)
        ax.set_xlabel(PATHWAY_LABELS.get(pathway_col, pathway_col), fontsize=8)
        ax.set_ylabel(PHENO_LABELS.get(pheno_col, pheno_col), fontsize=8)
        ax.set_title(f"Spearman {row['spearman_r']:.2f}", fontsize=9)
        ax.grid(alpha=0.12)
    fig.suptitle("Top pathway-phenotype target-level associations", fontweight="bold")
    fig.tight_layout()
    path = figs / "pathway_phenotype_top_pair_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_state_program_heatmap(table: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    cols = [column for column in PHENO_COLS + PATHWAY_COLS if column in table]
    state = table.groupby("state_class")[cols].mean(numeric_only=True)
    labels = [PHENO_LABELS.get(column, PATHWAY_LABELS.get(column, column)) for column in cols]
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    image = ax.imshow(state.to_numpy(dtype=float), cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(state.index)))
    ax.set_yticklabels(state.index, fontsize=8)
    for row_index in range(state.shape[0]):
        for col_index in range(state.shape[1]):
            value = state.iat[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("State-level phenotype and pathway program means")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = figs / "pathway_phenotype_state_program_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def render_html(
    out: Path,
    coords: pd.DataFrame,
    correlations: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    figures: dict[str, Path],
    target: pd.DataFrame,
) -> Path:
    state_counts = target["state_class"].value_counts().rename_axis("state_class").reset_index(name="n_targets")
    top = correlations.sort_values("abs_spearman_r", ascending=False).head(12)
    strongest_targets = target.sort_values(["consensus_score", "phenotype_strength"], ascending=False)[
        ["group", "state_class", "kd_tier", "tox_rate", "consensus_score", "phenotype_strength", "permito", "mitomass", "area", "intensity"]
    ].head(18)
    matrix_html = html_table(spearman_matrix.reset_index().rename(columns={"pathway_label": "pathway"}), max_rows=20)
    dino_section = ""
    dino_summary_path = out / "dino_state_separation_summary.csv"
    dino_nn_path = out / "dino_nearest_neighbors.csv"
    dino_figs = {
        "c1_tsne": out / "figs" / "dino_c1_tsne_state_map.png",
        "c24_tsne": out / "figs" / "dino_c24_tsne_state_map.png",
        "distance": out / "figs" / "dino_state_centroid_distance_heatmap.png",
    }
    if dino_summary_path.exists() and all(path.exists() for path in dino_figs.values()):
        dino_summary = pd.read_csv(dino_summary_path)
        dino_nn_html = "<p class='muted'>No nearest-neighbor table available.</p>"
        if dino_nn_path.exists():
            dino_nn = pd.read_csv(dino_nn_path)
            dino_nn = dino_nn[dino_nn["rank"].le(3)][
                ["channel", "query", "rank", "neighbor", "neighbor_state", "rms_distance"]
            ]
            dino_nn_html = html_table(dino_nn, max_rows=30)
        c24_row = dino_summary[dino_summary["channel"].eq("C24 mitochondrial")]
        c1_row = dino_summary[dino_summary["channel"].eq("C1 brightfield")]
        c24_nn = c24_row["nearest_neighbor_same_state_fraction"].iloc[0] if not c24_row.empty else np.nan
        c1_nn = c1_row["nearest_neighbor_same_state_fraction"].iloc[0] if not c1_row.empty else np.nan
        dino_section = f"""
  <section>
    <h2>DINO Feature Space</h2>
    <div class="callout">DINO state separation is weak globally, but C24 mitochondrial features are slightly more state-aware than C1 brightfield here: nearest-neighbor same-state fraction {c24_nn:.3f} vs {c1_nn:.3f}.</div>
    <div class="two-col">
      <figure>
        <img src="{relpath(dino_figs['c1_tsne'], out)}" alt="DINO C1 t-SNE">
        <figcaption>C1 brightfield DINOv2 target-centroid t-SNE, batch NTC-z standardized.</figcaption>
      </figure>
      <figure>
        <img src="{relpath(dino_figs['c24_tsne'], out)}" alt="DINO C24 t-SNE">
        <figcaption>C24 mitochondrial DINOv2 target-centroid t-SNE, batch NTC-z standardized.</figcaption>
      </figure>
    </div>
  </section>

  <section class="two-col">
    <div>
      <h2>DINO State Separation</h2>
      <div class="table-scroll">{html_table(dino_summary, max_rows=8)}</div>
    </div>
    <div>
      <h2>DINO Centroid Distances</h2>
      <figure>
        <img src="{relpath(dino_figs['distance'], out)}" alt="DINO state centroid distance heatmap">
        <figcaption>State centroid RMS distances in standardized DINO feature space.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2>DINO Nearest Neighbors</h2>
    <div class="table-scroll">{dino_nn_html}</div>
  </section>
"""
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pathway-Phenotype Correlation Atlas</title>
  <style>
    :root {{
      --ink: #18202b;
      --muted: #64748b;
      --line: #d8dee8;
      --panel: #ffffff;
      --bg: #f4f6f1;
      --accent: #0f766e;
      --accent-2: #b45309;
    }}
    body {{ margin: 0; font-family: Avenir Next, Noto Sans, Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 38px 48px 30px; background: linear-gradient(135deg, #f8fafc 0%, #e8efe7 56%, #f5eadb 100%); border-bottom: 1px solid var(--line); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 54px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0 0 12px; font-size: 17px; }}
    p {{ line-height: 1.62; }}
    .lead {{ max-width: 920px; color: #334155; font-size: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 20px; }}
    .metric {{ background: rgba(255,255,255,.72); border: 1px solid var(--line); padding: 14px 16px; }}
    .metric b {{ display: block; font-size: 24px; margin-bottom: 4px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    * {{ box-sizing: border-box; }}
    section {{ min-width: 0; background: var(--panel); border: 1px solid var(--line); margin: 18px 0; padding: 22px; box-shadow: 0 10px 28px rgba(31, 41, 55, .05); }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; display: block; border: 1px solid var(--line); background: #fff; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .two-col {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); gap: 18px; align-items: start; }}
    .two-col > *, .grid > * {{ min-width: 0; }}
    .table-scroll {{ max-width: 100%; min-width: 0; overflow-x: auto; border: 1px solid var(--line); }}
    table.data {{ border-collapse: collapse; width: max-content; min-width: 100%; max-width: none; font-size: 13px; }}
    table.data th, table.data td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 9px; text-align: right; white-space: nowrap; vertical-align: top; }}
    table.data th:first-child, table.data td:first-child {{ text-align: left; }}
    table.data td:nth-child(3), table.data td:last-child {{ max-width: 420px; white-space: normal; overflow-wrap: anywhere; }}
    table.data th {{ background: #f8fafc; color: #334155; position: sticky; top: 0; }}
    .callout {{ border-left: 4px solid var(--accent); background: #ecfdf5; padding: 13px 15px; margin: 14px 0; color: #134e4a; }}
    .files code {{ background: #f1f5f9; padding: 2px 5px; }}
    @media (max-width: 860px) {{
      header {{ padding: 26px 22px; }}
      .grid, .two-col {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px 14px 42px; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>Pathway-Phenotype Correlation Atlas</h1>
  <p class="lead">基于 EE DRUG-seq target-level pathway program 与 TMRM/MitoTracker phenotype 轴，展示相关性、t-SNE 结构和 state-level 机制轮廓。</p>
  <div class="grid">
    <div class="metric"><b>{len(target)}</b><span>Target-level perturbations</span></div>
    <div class="metric"><b>{len(PATHWAY_COLS)}</b><span>Curated pathway programs</span></div>
    <div class="metric"><b>{len(PHENO_COLS)}</b><span>Phenotype axes</span></div>
    <div class="metric"><b>{coords['tsne_perplexity'].iloc[0]}</b><span>t-SNE perplexity</span></div>
  </div>
</header>
<main>
  <section>
    <h2>核心读法</h2>
    <div class="callout">Pathway 与 phenotype 有方向性耦合，但不是强冗余。最高 Spearman 约 {top['spearman_r'].abs().max():.2f}，更适合作为 MoA 解释层，而不是单独的 phenotype predictor。</div>
    <p><b>Mito biogenesis</b> 与 per-mito dPsi / TMRM intensity 正相关，但与 MitoTracker mass 负相关，提示影像上的 mass-like 状态不等于经典生物发生转录程序。<b>OXPHOS/ETC</b> 与 TMRM collapse 方向负相关，同时弱正相关于 MitoTracker mass，更像补偿性 remodeling。</p>
  </section>

  <section>
    <h2>t-SNE：Pathway + Phenotype Target Map</h2>
    <figure>
      <img src="{relpath(figures['tsne'], out)}" alt="pathway phenotype t-SNE">
      <figcaption>输入特征为 7 个 pathway score + 4 个 phenotype 轴，StandardScaler 后计算 t-SNE；颜色为 MoA state，点大小随 phenotype_strength 增大。</figcaption>
    </figure>
  </section>

{dino_section}

  <section class="two-col">
    <div>
      <h2>Spearman Heatmap</h2>
      <figure>
        <img src="{relpath(figures['spearman'], out)}" alt="spearman heatmap">
        <figcaption>推荐主读数：target-level Spearman，适合看排序一致性。</figcaption>
      </figure>
    </div>
    <div>
      <h2>State Counts</h2>
      <div class="table-scroll">{html_table(state_counts, max_rows=12)}</div>
    </div>
  </section>

  <section>
    <h2>Top Pair Scatter</h2>
    <figure>
      <img src="{relpath(figures['pairs'], out)}" alt="top pathway phenotype scatter">
      <figcaption>前 6 个绝对 Spearman 最高的 pathway-phenotype pair；颜色同 t-SNE state。</figcaption>
    </figure>
  </section>

  <section>
    <h2>State-Level Program Profile</h2>
    <figure>
      <img src="{relpath(figures['state'], out)}" alt="state program heatmap">
      <figcaption>各 MoA state 的 phenotype 与 pathway program 均值，可辅助判断 toxic collapse、uncoupler-like、biogenesis-like 是否有一致机制轮廓。</figcaption>
    </figure>
  </section>

  <section class="two-col">
    <div>
      <h2>最强相关</h2>
      <div class="table-scroll">{html_table(top[['pathway_label', 'phenotype_label', 'n_targets', 'spearman_r', 'spearman_fdr', 'pearson_r']], max_rows=12)}</div>
    </div>
    <div>
      <h2>Spearman Matrix</h2>
      <div class="table-scroll">{matrix_html}</div>
    </div>
  </section>

  <section>
    <h2>High-Signal Targets</h2>
    <div class="table-scroll">{html_table(strongest_targets, max_rows=18)}</div>
  </section>

  <section class="files">
    <h2>Files</h2>
    <p>HTML source uses <code>pathway_phenotype_correlations_long.csv</code>, <code>pathway_phenotype_spearman_matrix.csv</code>, and <code>crossmodal_moa_target_table.csv</code>.</p>
    <p>DINO source files, when present: <code>dino_state_separation_summary.csv</code>, <code>dino_nearest_neighbors.csv</code>, and <code>figs/dino_c1_tsne_state_map.png</code>/<code>figs/dino_c24_tsne_state_map.png</code>.</p>
    <p>New outputs: <code>pathway_phenotype_visualization.html</code>, <code>target_pathway_phenotype_tsne.csv</code>, and the figure PNGs under <code>figs/</code>.</p>
  </section>
</main>
</body>
</html>
"""
    path = out / "pathway_phenotype_visualization.html"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args()
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    target_table = Path(args.target_table) if args.target_table else out / "crossmodal_moa_target_table.csv"
    correlations_long = Path(args.correlations_long) if args.correlations_long else out / "pathway_phenotype_correlations_long.csv"
    spearman_matrix_path = Path(args.spearman_matrix) if args.spearman_matrix else out / "pathway_phenotype_spearman_matrix.csv"

    table = pd.read_csv(target_table)
    target = table[table["category"].eq("Target")].copy()
    correlations = pd.read_csv(correlations_long)
    spearman_matrix = pd.read_csv(spearman_matrix_path).rename(columns={"Unnamed: 0": "pathway_label"})
    if "pathway_label" not in spearman_matrix:
        spearman_matrix = spearman_matrix.rename(columns={spearman_matrix.columns[0]: "pathway_label"})

    coords = compute_tsne(target, PATHWAY_COLS + PHENO_COLS, args.random_state)
    coords.to_csv(out / "target_pathway_phenotype_tsne.csv", index=False)

    figures = {
        "tsne": plot_tsne(coords, out),
        "spearman": figs / "pathway_phenotype_spearman_heatmap.png",
        "pairs": plot_top_pair_scatter(target, correlations, out),
        "state": plot_state_program_heatmap(target, out),
    }
    report = render_html(out, coords, correlations, spearman_matrix, figures, target)
    print(f"[pathway_phenotype_visualization] wrote {report}")
    print(f"[pathway_phenotype_visualization] wrote {out / 'target_pathway_phenotype_tsne.csv'}")
    print(f"[pathway_phenotype_visualization] figures under {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())