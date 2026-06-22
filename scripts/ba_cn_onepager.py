#!/usr/bin/env python
"""Generate a Chinese one-pager and readable figures for the B+A MoA story."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


STATE_ORDER = [
    "neutral_or_uncertain",
    "uncoupler_like",
    "mixed_uncoupling_biogenesis",
    "biogenesis_like",
    "toxic_collapse",
]

STATE_SHORT = {
    "neutral_or_uncertain": "Neutral",
    "uncoupler_like": "Uncoupler",
    "mixed_uncoupling_biogenesis": "Mixed",
    "biogenesis_like": "Bio-like",
    "toxic_collapse": "Toxic",
}

STATE_COLORS = {
    "neutral_or_uncertain": "#8c8c8c",
    "uncoupler_like": "#d95f02",
    "mixed_uncoupling_biogenesis": "#7570b3",
    "biogenesis_like": "#1b9e77",
    "toxic_collapse": "#b2182b",
}

FEATURE_LABELS = {
    "permito": "per-mito\ndPsi",
    "mitomass": "mito\nmass",
    "area": "TMRM\narea",
    "tox_rate": "tox\nrate",
    "conn_PSMC3": "PSMC3\nconn.",
    "path_OXPHOS_ETC": "OXPHOS",
    "path_MITO_BIOGENESIS": "mito bio.\ngenes",
    "path_PROTEOSTASIS_AUTOPHAGY": "proteostasis",
    "path_APOPTOSIS_TOXICITY": "apoptosis",
}

REPRESENTATIVE_TARGETS = [
    "BAM15", "MK8722", "PSMC3", "DDI2", "G6PC", "TM6SF2",
    "DGAT2", "SLC39A11", "TAGLN", "NOTCH2", "SLC12A8", "INO80E",
]


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(args.table)
    tests = pd.read_csv(args.tests)
    volcano = pd.read_csv(args.volcano)
    return table, tests, volcano


def get_fdr(tests: pd.DataFrame, contrast: str, feature: str) -> float | None:
    sub = tests[(tests["contrast"] == contrast) & (tests["feature"] == feature)]
    if sub.empty:
        return None
    val = sub["fdr"].iloc[0]
    return float(val) if np.isfinite(val) else None


def format_fdr(value: float | None) -> str:
    if value is None:
        return "FDR n/a"
    if value < 1e-4:
        return f"FDR={value:.1e}"
    return f"FDR={value:.3f}"


def plot_readable_state_effects(table: pd.DataFrame, tests: pd.DataFrame, out: Path) -> None:
    target = table[table["category"].eq("Target")].copy()
    features = [
        "permito", "area", "mitomass", "tox_rate",
        "path_OXPHOS_ETC", "path_MITO_BIOGENESIS",
        "path_PROTEOSTASIS_AUTOPHAGY", "path_APOPTOSIS_TOXICITY",
    ]
    fdr_map = {
        "permito": get_fdr(tests, "uncoupling_states_vs_neutral", "permito"),
        "area": get_fdr(tests, "uncoupling_states_vs_neutral", "area"),
        "mitomass": get_fdr(tests, "uncoupling_states_vs_neutral", "mitomass"),
        "tox_rate": get_fdr(tests, "toxic_collapse_vs_all_other_states", "tox_rate"),
        "path_OXPHOS_ETC": get_fdr(tests, "uncoupling_states_vs_neutral", "path_OXPHOS_ETC"),
        "path_MITO_BIOGENESIS": get_fdr(tests, "biogenesis_like_vs_neutral", "path_MITO_BIOGENESIS"),
        "path_PROTEOSTASIS_AUTOPHAGY": get_fdr(tests, "toxic_collapse_vs_all_other_states", "path_PROTEOSTASIS_AUTOPHAGY"),
        "path_APOPTOSIS_TOXICITY": get_fdr(tests, "toxic_collapse_vs_all_other_states", "path_APOPTOSIS_TOXICITY"),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True)
    axes = axes.ravel()
    states = [state for state in STATE_ORDER if state in set(target["state_class"])]
    rng = np.random.default_rng(0)
    for ax, feature in zip(axes, features):
        values = [target[target["state_class"].eq(state)][feature].dropna().to_numpy(float) for state in states]
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        for patch, state in zip(bp["boxes"], states):
            patch.set_facecolor(STATE_COLORS.get(state, "#cccccc"))
            patch.set_alpha(0.72)
            patch.set_linewidth(1.2)
        for state_idx, state in enumerate(states, start=1):
            y = target[target["state_class"].eq(state)][feature].dropna().to_numpy(float)
            x = np.full(len(y), state_idx) + rng.normal(0, 0.04, size=len(y))
            ax.scatter(x, y, s=13, color="#222222", alpha=0.32, linewidth=0)
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_title(FEATURE_LABELS.get(feature, feature).replace("\n", " "), fontsize=13, weight="bold")
        ax.text(0.02, 0.96, format_fdr(fdr_map.get(feature)), transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9})
        ax.set_xticks(np.arange(1, len(states) + 1))
        ax.set_xticklabels([STATE_SHORT[state] for state in states], fontsize=10, rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("State-level MoA evidence (readable summary)", fontsize=18, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_readable_evidence(table: pd.DataFrame, out: Path) -> None:
    reps = table[table["group"].isin(REPRESENTATIVE_TARGETS)].copy()
    order = [target for target in REPRESENTATIVE_TARGETS if target in set(reps["group"])]
    reps["_order"] = reps["group"].map({target: i for i, target in enumerate(order)})
    reps = reps.sort_values("_order")
    features = [
        "permito", "mitomass", "area", "tox_rate",
        "conn_BAM15", "conn_MK8722", "conn_PSMC3",
        "path_OXPHOS_ETC", "path_MITO_BIOGENESIS",
        "path_PROTEOSTASIS_AUTOPHAGY", "path_APOPTOSIS_TOXICITY",
    ]
    labels = [
        "per-mito\ndPsi", "mito\nmass", "TMRM\narea", "tox\nrate",
        "BAM15\nconn.", "MK8722\nconn.", "PSMC3\nconn.",
        "OXPHOS", "mito bio.\ngenes", "proteostasis", "apoptosis",
    ]
    data = reps.set_index("group")[features].astype(float)
    scaled = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, np.nan)
    scaled = scaled.clip(-2.5, 2.5)
    fig, ax = plt.subplots(figsize=(17.5, 7.8))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(labels, fontsize=11)
    row_labels = [f"{idx}  [{STATE_SHORT.get(reps.set_index('group').loc[idx, 'state_class'], 'state')}]" for idx in data.index]
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Representative target evidence strips (larger, readable)", fontsize=17, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="column z-score")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return PCA(n_components=2, random_state=0).fit_transform(matrix)


def zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    return (out - out.mean(axis=0)) / out.std(axis=0).replace(0, np.nan)


def plot_readable_moa_map(table: pd.DataFrame, out: Path) -> None:
    path_cols = [col for col in table.columns if col.startswith("path_")]
    feature_cols = ["permito", "mitomass", "area", "intensity", "conn_BAM15", "conn_MK8722", "conn_PSMC3", "conn_ATP5B", "conn_SLC25A4"] + path_cols
    plot_df = table[table["category"].isin(["Target", "PC"])].copy()
    emb = pca_2d(zscore_cols(plot_df, feature_cols).to_numpy())
    plot_df["pc1"] = emb[:, 0]
    plot_df["pc2"] = emb[:, 1]
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    centroid_offsets = {
      "neutral_or_uncertain": (-18, 12),
      "uncoupler_like": (10, 14),
      "mixed_uncoupling_biogenesis": (12, -16),
      "biogenesis_like": (10, 16),
      "toxic_collapse": (12, 12),
    }
    for state in STATE_ORDER:
        sub = plot_df[plot_df["state_class"].eq(state)]
        if sub.empty:
            continue
        ax.scatter(sub["pc1"], sub["pc2"], s=45, alpha=0.62, color=STATE_COLORS[state], label=STATE_SHORT[state], edgecolor="white", linewidth=0.4)
        centroid = sub[["pc1", "pc2"]].mean()
        ax.scatter(centroid["pc1"], centroid["pc2"], s=220, color=STATE_COLORS[state], edgecolor="black", linewidth=1.0)
        ax.annotate(
          STATE_SHORT[state],
          (centroid["pc1"], centroid["pc2"]),
          xytext=centroid_offsets.get(state, (8, 8)),
          textcoords="offset points",
          fontsize=12,
          weight="bold",
          bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    for group in ["BAM15", "MK8722", "PSMC3", "DDI2", "TAGLN", "TM6SF2", "DGAT2", "G6PC"]:
        sub = plot_df[plot_df["group"].eq(group)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        ax.annotate(group, (row["pc1"], row["pc2"]), xytext=(5, -9), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.axvline(0, color="#888888", lw=0.7)
    ax.set_xlabel("Cross-modal MoA PC1", fontsize=12)
    ax.set_ylabel("Cross-modal MoA PC2", fontsize=12)
    ax.set_title("Cross-modal MoA map (state centroids + selected targets)", fontsize=17, weight="bold")
    ax.legend(fontsize=10, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_readable_volcano(volcano: pd.DataFrame, out: Path) -> None:
    contrasts = [
        "uncoupling_states_vs_neutral",
        "biogenesis_like_vs_neutral",
        "toxic_collapse_vs_neutral",
    ]
    titles = {
        "uncoupling_states_vs_neutral": "Uncoupling states vs neutral",
        "biogenesis_like_vs_neutral": "Biogenesis-like vs neutral",
        "toxic_collapse_vs_neutral": "Toxic-collapse vs neutral",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    for ax, contrast in zip(axes, contrasts):
        sub = volcano[volcano["contrast"].eq(contrast)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = np.where(
            sub["significant"] & (sub["effect_ntc_z"] > 0),
            "#b2182b",
            np.where(sub["significant"] & (sub["effect_ntc_z"] < 0), "#2166ac", "#bdbdbd"),
        )
        ax.scatter(sub["effect_ntc_z"], sub["neg_log10_fdr"], s=13, c=colors, alpha=0.72, linewidth=0)
        pathway = sub[sub["curated_pathway_gene"]]
        ax.scatter(pathway["effect_ntc_z"], pathway["neg_log10_fdr"], s=34, facecolors="none", edgecolors="#fdae61", linewidth=0.9)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.axvline(1, color="#999999", lw=0.6, ls="--")
        ax.axvline(-1, color="#999999", lw=0.6, ls="--")
        ax.axhline(-np.log10(0.1), color="#999999", lw=0.6, ls="--")
        labels = sub.assign(label_score=sub["effect_ntc_z"].abs() * sub["neg_log10_fdr"].fillna(0))
        labels = labels.sort_values("label_score", ascending=False).head(4).reset_index(drop=True)
        offsets = [(5, 7), (5, -12), (-42, 7), (-42, -12)]
        for i, row in labels.iterrows():
          ax.annotate(
            row["symbol"],
            (row["effect_ntc_z"], row["neg_log10_fdr"]),
            xytext=offsets[i % len(offsets)],
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
          )
        ax.set_title(titles[contrast], fontsize=13, weight="bold")
        ax.set_xlabel("mean state effect vs neutral", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
    axes[0].set_ylabel("-log10(FDR)", fontsize=11)
    fig.suptitle("Transcriptomic state volcano plots (readable)", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_cn_html(out_dir: Path) -> None:
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B+A 跨模态 MoA 中文 One-pager</title>
  <style>
    :root { --ink:#17212b; --muted:#5d6b78; --line:#d9e1e8; --paper:#f7f9fb; --panel:#fff; --blue:#0b4f7a; --green:#347a3d; --orange:#b45619; --red:#a83232; --purple:#67518a; --amber:#8a650e; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font-family:"Aptos","Segoe UI","Noto Sans SC",sans-serif; line-height:1.5; }
    main { width:min(1240px, calc(100vw - 32px)); margin:22px auto 40px; }
    header, section { background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 22px rgba(23,33,43,.05); }
    header { padding:24px 28px; margin-bottom:16px; }
    section { padding:18px; margin-bottom:16px; }
    h1 { margin:0 0 10px; font-size:34px; line-height:1.12; letter-spacing:0; }
    h2 { margin:0 0 12px; color:var(--blue); font-size:22px; }
    h3 { margin:0 0 8px; font-size:17px; }
    p { margin:0 0 10px; }
    .lead { font-size:18px; max-width:80ch; color:#263747; }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
    .metric { border:1px solid var(--line); border-radius:7px; padding:12px; background:#fbfdff; }
    .metric strong { display:block; font-size:28px; color:var(--blue); line-height:1.1; }
    .metric span { color:var(--muted); font-size:13px; }
    .tag { display:inline-block; padding:3px 8px; border-radius:999px; color:#fff; font-size:12px; font-weight:700; margin:0 5px 6px 0; }
    .red{background:var(--red)} .orange{background:var(--orange)} .green{background:var(--green)} .blue{background:var(--blue)} .purple{background:var(--purple)} .amber{background:var(--amber)}
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th,td { text-align:left; vertical-align:top; padding:8px 7px; border-bottom:1px solid var(--line); }
    th { color:var(--blue); background:#f0f5f8; }
    ul,ol { margin:8px 0 0; padding-left:20px; }
    li { margin:6px 0; }
    .callout { border-left:5px solid var(--blue); background:#eef6fb; padding:12px 14px; border-radius:6px; margin-top:10px; }
    .warn { border-left-color:var(--red); background:#fff2f2; }
    .figs { display:grid; grid-template-columns:1fr; gap:18px; }
    figure { margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }
    figure img { display:block; width:100%; max-height:760px; object-fit:contain; background:#fff; border-bottom:1px solid var(--line); }
    figcaption { padding:10px 12px; color:var(--muted); font-size:13px; }
    a { color:var(--blue); font-weight:700; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .small { color:var(--muted); font-size:13px; }
    @media (max-width:900px) { .grid-2,.grid-3,.metrics { grid-template-columns:1fr; } h1 { font-size:28px; } }
    @media print { body{background:#fff} main{width:auto;margin:0} header,section{box-shadow:none;break-inside:avoid} figure img{max-height:540px} }
  </style>
</head>
<body>
<main>
  <header>
    <p class="small">B+A cross-modal MoA · 中文 one-pager · 2026-06-22</p>
    <h1>目前故事：EE 扰动形成跨模态线粒体 MoA 状态</h1>
    <p class="lead">不要把结果讲成单纯 hit prioritization。更好的故事是：影像定义线粒体功能状态，DRUG-seq 负责审稿和改写机制解释，把 clean remodeling、toxicity artifact、KD failed 和 noncanonical imaging state 分开。</p>
    <div class="metrics">
      <div class="metric"><strong>175</strong><span>targets scored</span></div>
      <div class="metric"><strong>69</strong><span>imaging-strong hits</span></div>
      <div class="metric"><strong>23</strong><span>hard kill / deprioritize</span></div>
      <div class="metric"><strong>21</strong><span>repeat / reinterpret</span></div>
    </div>
  </header>

  <div class="grid-3">
    <section>
      <h2>故事 1</h2>
      <h3>Uncoupling 不是单轴 TMRM 下降</h3>
      <p>uncoupler-like 和 mixed states 相对 neutral 有显著 per-mito dPsi 下降、TMRM area 下降；mixed state 还叠加 mito mass 上升。</p>
      <span class="tag orange">dPsi down</span><span class="tag orange">area down</span><span class="tag purple">mass up</span>
      <ul><li>per-mito dPsi FDR ~ 6.5e-23</li><li>TMRM area FDR ~ 8.7e-14</li><li>OXPHOS 上升，canonical mito-biogenesis score 下降</li></ul>
    </section>
    <section>
      <h2>故事 2</h2>
      <h3>Toxic-collapse 是独立 confounder</h3>
      <p>toxic-collapse 不只是更强的 uncoupling。它有 cell-loss、更深 area collapse、更强 per-mito dPsi loss 和显著 apoptosis/toxicity program。</p>
      <span class="tag red">toxicity</span><span class="tag red">apoptosis</span><span class="tag amber">PSMC3 trend</span>
      <ul><li>tox_rate FDR ~ 1.1e-13</li><li>apoptosis/toxicity FDR ~ 0.014</li><li>PSMC3/proteostasis 只作为方向性支持，不做主 claim</li></ul>
    </section>
    <section>
      <h2>故事 3</h2>
      <h3>Biogenesis-like 不是经典 PGC1A/TFAM</h3>
      <p>MitoTracker mass 很强，但 curated mitochondrial-biogenesis transcript score 反而下降。这提示非经典 remodeling、形态/染料摄取或膜结构变化。</p>
      <span class="tag green">mass up</span><span class="tag blue">noncanonical</span>
      <ul><li>mito mass FDR ~ 6.4e-8</li><li>mito-biogenesis genes lower, FDR ~ 0.014</li><li>这是一个可讲的反直觉发现</li></ul>
    </section>
  </div>

  <section>
    <h2>DRUG-seq 审稿 playbook：哪些 imaging hit 会被枪毙/改写</h2>
    <table>
      <thead><tr><th>结论</th><th>规则</th><th>代表靶点</th><th>推荐措辞</th></tr></thead>
      <tbody>
        <tr><td><span class="tag red">Hard kill</span></td><td>KD failed；tox_rate >= 0.3；toxic-collapse；apoptosis + proteostasis 高；或 PSMC3-like + proteostasis。</td><td>PSMC3, KANSL1, CHRNE, RFT1, RPL8, CDK2AP1, C8orf58, C2orf16, BIN3, CCAR2, FARSA, PNPLA3, SCAF1, FADS3, LTBP3, SHISA5, MYO19, VPS11, INO80E, OSGIN1, SERPINA1, SRRM2.</td><td>影像表型存在，但不是干净 EE target mechanism。</td></tr>
        <tr><td><span class="tag amber">Repeat / reinterpret</span></td><td>KD unknown 或跨模态 mismatch。保留生物学现象，但不能照原 imaging 解释讲。</td><td>DDI2, TAGLN, SLC9B2, TRIM63, MRC2, ANGPTL7, HIST1H4H, PM20D1, GTF3A, ANGEL1, ADORA2A, ELP6, SIRPB1, KREMEN1, SLC12A2, ZNF654, ARFGAP3, MYO5C, ATE1, CYP21A2, NFAT5.</td><td>DRUG-seq 改写机制标签，不一定否定 target 价值。</td></tr>
        <tr><td><span class="tag green">Keep as story hits</span></td><td>影像强、KD 可信、tox 低、无明显 stress program，并能放进一致 MoA state。</td><td>TM6SF2, SLC39A11, DGAT2, NOTCH2, SLC12A8, TCF12, DCAKD, TIPIN, ALDH1A2.</td><td>用于锚定 MoA states 和 representative evidence strips。</td></tr>
      </tbody>
    </table>
    <div class="callout warn"><strong>注意：</strong>G6PC 不是 KD/toxicity 意义上的 hard kill，但有 apoptosis/proteostasis 和 OXPHOS mismatch。建议讲成 stress-linked uncoupler-like phenotype，而不是 clean uncoupler hit。</div>
  </section>

  <section>
    <h2>优化后的主图</h2>
    <div class="figs">
      <figure><a href="figs/cn_state_effects_readable.png"><img src="figs/cn_state_effects_readable.png" alt="state effects"></a><figcaption>状态层面的核心证据：大字号 boxplots + FDR callouts。</figcaption></figure>
      <figure><a href="figs/cn_crossmodal_moa_map_readable.png"><img src="figs/cn_crossmodal_moa_map_readable.png" alt="moa map"></a><figcaption>跨模态 MoA map：显示 state centroids 和少数代表靶点，避免标签拥挤。</figcaption></figure>
      <figure><a href="figs/cn_representative_evidence_readable.png"><img src="figs/cn_representative_evidence_readable.png" alt="evidence strips"></a><figcaption>代表靶点 evidence strips：用于从全局状态落到具体 target。</figcaption></figure>
      <figure><a href="figs/cn_transcriptomic_volcano_readable.png"><img src="figs/cn_transcriptomic_volcano_readable.png" alt="volcano"></a><figcaption>state-vs-neutral transcriptomic volcano：更大字号、更少标签。</figcaption></figure>
    </div>
  </section>

  <div class="grid-2">
    <section>
      <h2>最适合讲的代表靶点</h2>
      <table>
        <thead><tr><th>角色</th><th>靶点</th><th>用途</th></tr></thead>
        <tbody>
          <tr><td>Controls</td><td>BAM15, MK8722, PSMC3</td><td>定义 uncoupling、mass/energizer、toxic-collapse anchor。</td></tr>
          <tr><td>Mixed states</td><td>TM6SF2, DGAT2, SLC39A11, NOTCH2</td><td>证明 potential 和 mass 必须拆开。</td></tr>
          <tr><td>Reinterpreted hits</td><td>DDI2, TAGLN, G6PC</td><td>展示 DRUG-seq 如何阻止过度解释。</td></tr>
          <tr><td>Clean-ish retains</td><td>SLC12A8, TCF12, DCAKD, TIPIN</td><td>后续讨论或补充验证候选。</td></tr>
        </tbody>
      </table>
    </section>
    <section>
      <h2>下一步</h2>
      <ol>
        <li>把 readable figures 组合成一页主图草稿。</li>
        <li>给 boxplot panel 标 FDR 星号/括号，做成 manuscript 风格。</li>
        <li>把 prioritization 放 supplement，主文只讲 MoA state discovery。</li>
        <li>如果要验证，优先围绕“mixed state”和“noncanonical biogenesis-like state”设计，而不是泛泛验证 top hits。</li>
      </ol>
    </section>
  </div>

  <section>
    <h2>源文件</h2>
    <p><a href="B_A_crossmodal_moa_story.md">story markdown</a> · <a href="state_moa_tests.csv">state tests</a> · <a href="crossmodal_moa_target_table.csv">target MoA table</a> · <a href="transcriptomic_state_volcano.csv">volcano stats</a> · <a href="representative_target_evidence.csv">representative evidence</a></p>
    <p class="small">建议打开本 HTML 后直接浏览器打印成 PDF。</p>
  </section>
</main>
</body>
</html>
"""
    (out_dir / "B_A_crossmodal_moa_one_pager_CN.html").write_text(html)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--tests", default="output/2026-06-22/ba_multimodal_plan/state_moa_tests.csv")
    parser.add_argument("--volcano", default="output/2026-06-22/ba_multimodal_plan/transcriptomic_state_volcano.csv")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    table, tests, volcano = load_inputs(args)

    plot_readable_state_effects(table, tests, figs / "cn_state_effects_readable.png")
    plot_readable_evidence(table, figs / "cn_representative_evidence_readable.png")
    plot_readable_moa_map(table, figs / "cn_crossmodal_moa_map_readable.png")
    plot_readable_volcano(volcano, figs / "cn_transcriptomic_volcano_readable.png")
    write_cn_html(out_dir)
    print(f"[ba_cn_onepager] wrote {out_dir / 'B_A_crossmodal_moa_one_pager_CN.html'}")
    print(f"[ba_cn_onepager] wrote readable figures under {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())