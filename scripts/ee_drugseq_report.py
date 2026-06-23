#!/usr/bin/env python
"""Generate a dedicated EE DRUG-seq HTML analysis report.

The report focuses on the DRUG-seq data layer: assay QC, perturbation
efficiency, transcriptomic state programs, and target-level signatures. Imaging
phenotypes are used only as matched reference axes for interpretation.

Run with the scvi/anndata environment:

    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/ee_drugseq_report.py
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


STATE_ORDER = [
    "neutral_or_uncertain",
    "uncoupler_like",
    "mixed_uncoupling_biogenesis",
    "biogenesis_like",
    "toxic_collapse",
]

STATE_LABELS = {
    "neutral_or_uncertain": "Neutral / uncertain",
    "uncoupler_like": "Uncoupler-like",
    "mixed_uncoupling_biogenesis": "Mixed uncoupling + biogenesis",
    "biogenesis_like": "Biogenesis-like",
    "toxic_collapse": "Toxic collapse",
}

STATE_COLORS = {
    "neutral_or_uncertain": "#8b9097",
    "uncoupler_like": "#1f77b4",
    "mixed_uncoupling_biogenesis": "#2ca25f",
    "biogenesis_like": "#d98c21",
    "toxic_collapse": "#c43b3b",
}

PROGRAM_COLS = [
    "path_OXPHOS_ETC",
    "path_MITO_BIOGENESIS",
    "path_FAO_LIPID",
    "path_AMPK_MTOR_INSULIN",
    "path_ISR_ER_STRESS",
    "path_PROTEOSTASIS_AUTOPHAGY",
    "path_APOPTOSIS_TOXICITY",
]

PROGRAM_LABELS = {
    "path_OXPHOS_ETC": "OXPHOS / ETC",
    "path_MITO_BIOGENESIS": "Mito biogenesis",
    "path_FAO_LIPID": "FAO / lipid",
    "path_AMPK_MTOR_INSULIN": "AMPK-mTOR-insulin",
    "path_ISR_ER_STRESS": "ISR / ER stress",
    "path_PROTEOSTASIS_AUTOPHAGY": "Proteostasis / autophagy",
    "path_APOPTOSIS_TOXICITY": "Apoptosis / toxicity",
}

PHENO_LABELS = {
    "permito": "per-mito dPsi",
    "mitomass": "mito mass",
    "area": "TMRM area",
    "intensity": "TMRM intensity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--adata", default="data/processed/adata_drugseq_processed.h5ad")
    parser.add_argument("--target-table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--correlations", default="output/2026-06-22/ba_multimodal_plan/pathway_phenotype_correlations.csv")
    parser.add_argument("--state-tests", default="output/2026-06-22/ba_multimodal_plan/state_moa_tests.csv")
    parser.add_argument("--out", default="output/2026-06-23/drugseq_report")
    parser.add_argument("--min-target-wells", type=int, default=4)
    return parser.parse_args()


def as_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)


def pca_scores(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    return matrix @ vt[:n_components].T / np.maximum(singular_values[:n_components], 1e-12)


def safe_state_order(states: pd.Series) -> list[str]:
    observed = [state for state in STATE_ORDER if state in set(states.dropna())]
    extra = sorted(set(states.dropna()) - set(observed))
    return observed + extra


def html_table(df: pd.DataFrame, *, max_rows: int = 25, float_format: str = "{:.3f}") -> str:
    if df.empty:
        return "<p class='muted'>无可用数据。</p>"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else float_format.format(float(value)))
    return view.to_html(index=False, escape=True, classes="data")


def figure_path(path: Path, out: Path) -> str:
    return html.escape(path.relative_to(out).as_posix())


def make_qc_pca(adata, out: Path) -> Path:
    figs = out / "figs"
    raw = pca_scores(as_dense(adata.obsm["X_lognorm_hvg"]))
    corrected = pca_scores(as_dense(adata.obsm["X_zscore_hvg"]))
    obs = adata.obs.copy()
    plates = obs["plate"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.3))
    for ax, embedding, title in [
        (axes[0], raw, "Before within-NTC standardization"),
        (axes[1], corrected, "After within-NTC standardization"),
    ]:
        for plate in sorted(pd.unique(plates)):
            mask = plates == plate
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.58, label=plate)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.15)
    axes[1].legend(title="plate", fontsize=7, title_fontsize=8, ncol=2, frameon=False)
    fig.suptitle("DRUG-seq QC: batch structure before and after correction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = figs / "drugseq_qc_batch_pca.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def target_expression_centroids(adata, target_table: pd.DataFrame, min_target_wells: int) -> tuple[pd.DataFrame, np.ndarray]:
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["category"] = obs["category"].astype(str)
    keep = np.ones(len(obs), dtype=bool)
    if "qc_fail" in obs:
        keep &= ~obs["qc_fail"].astype(bool).to_numpy()
    X = as_dense(adata.obsm["X_zscore_hvg"])[keep]
    obs_keep = obs.loc[keep].copy().reset_index(drop=True)
    obs_keep["_row"] = np.arange(len(obs_keep))

    rows = []
    deltas = []
    for group, sub in obs_keep.groupby("group", sort=True):
        category = str(sub["category"].mode().iloc[0])
        if category != "Target" or len(sub) < min_target_wells:
            continue
        idx = sub["_row"].to_numpy(dtype=int)
        rows.append({
            "group": group,
            "n_wells_expression": int(len(sub)),
            "tox_rate_well": float(sub["tox_flag"].astype(bool).mean()) if "tox_flag" in sub else np.nan,
        })
        deltas.append(X[idx].mean(axis=0))

    meta = pd.DataFrame(rows)
    cols = [
        "group",
        "state_class",
        "recommendation",
        "kd_tier",
        "tox_rate",
        "consensus_score",
        "permito",
        "mitomass",
        "area",
    ] + [col for col in PROGRAM_COLS if col in target_table]
    meta = meta.merge(target_table[cols].drop_duplicates("group"), on="group", how="left")
    return meta, np.vstack(deltas)


def make_target_pca(adata, target_table: pd.DataFrame, out: Path, min_target_wells: int) -> tuple[Path, pd.DataFrame]:
    figs = out / "figs"
    meta, deltas = target_expression_centroids(adata, target_table, min_target_wells)
    coords = pca_scores(deltas)
    meta["drugseq_pc1"] = coords[:, 0]
    meta["drugseq_pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(11.2, 8.0))
    for state in safe_state_order(meta["state_class"]):
        mask = meta["state_class"].eq(state)
        ax.scatter(
            meta.loc[mask, "drugseq_pc1"],
            meta.loc[mask, "drugseq_pc2"],
            s=52,
            alpha=0.82,
            label=STATE_LABELS.get(state, state),
            color=STATE_COLORS.get(state, "#555555"),
            edgecolor="white",
            linewidth=0.5,
        )
    label_df = pd.concat([
        meta.nlargest(3, "consensus_score"),
        meta[meta["state_class"].eq("toxic_collapse")],
    ]).drop_duplicates("group")
    for _, row in label_df.iterrows():
        ax.text(row["drugseq_pc1"], row["drugseq_pc2"], str(row["group"]), fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#9aa0a6", lw=0.7)
    ax.axvline(0, color="#9aa0a6", lw=0.7)
    ax.set_xlabel("DRUG-seq target signature PC1")
    ax.set_ylabel("DRUG-seq target signature PC2")
    ax.set_title("Target-level DRUG-seq signature space", fontweight="bold")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path = figs / "drugseq_target_signature_pca.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, meta


def make_state_count_plot(target_table: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    data = target_table[target_table["category"].eq("Target")].copy()
    state_order = safe_state_order(data["state_class"])
    counts = data["state_class"].value_counts().reindex(state_order).fillna(0)
    kd = pd.crosstab(data["state_class"], data["kd_tier"]).reindex(state_order).fillna(0)
    kd_cols = [col for col in ["strong", "weak", "failed", "unknown"] if col in kd.columns] + [
        col for col in kd.columns if col not in {"strong", "weak", "failed", "unknown"}
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), gridspec_kw={"width_ratios": [1, 1.2]})
    labels = [STATE_LABELS.get(state, state) for state in counts.index]
    axes[0].barh(labels, counts.to_numpy(), color=[STATE_COLORS.get(state, "#666666") for state in counts.index])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("# targets")
    axes[0].set_title("MoA state distribution")
    left = np.zeros(len(kd))
    kd_palette = {"strong": "#176d4d", "weak": "#8fbf6f", "failed": "#d95f59", "unknown": "#b4b7bb"}
    for col in kd_cols:
        axes[1].barh(labels, kd[col].to_numpy(), left=left, label=col, color=kd_palette.get(col, "#777777"))
        left += kd[col].to_numpy()
    axes[1].invert_yaxis()
    axes[1].set_xlabel("# targets")
    axes[1].set_title("Knockdown tier by state")
    axes[1].legend(title="KD tier", frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="x", alpha=0.16)
    fig.suptitle("DRUG-seq target state and perturbation quality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = figs / "drugseq_state_counts_kd.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_program_state_heatmap(target_table: pd.DataFrame, out: Path) -> tuple[Path, pd.DataFrame]:
    figs = out / "figs"
    data = target_table[target_table["category"].eq("Target")].copy()
    state_order = safe_state_order(data["state_class"])
    available_programs = [col for col in PROGRAM_COLS if col in data.columns]
    summary = data.groupby("state_class")[available_programs].mean().reindex(state_order)

    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    max_abs = float(np.nanmax(np.abs(summary.to_numpy()))) if not summary.empty else 1.0
    max_abs = max(max_abs, 0.25)
    image = ax.imshow(summary.to_numpy(), cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_yticks(np.arange(len(summary.index)))
    ax.set_yticklabels([STATE_LABELS.get(state, state) for state in summary.index])
    ax.set_xticks(np.arange(len(available_programs)))
    ax.set_xticklabels([PROGRAM_LABELS[col] for col in available_programs], rotation=35, ha="right")
    for y in range(summary.shape[0]):
        for x in range(summary.shape[1]):
            value = summary.iloc[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("mean DRUG-seq program score")
    ax.set_title("Transcriptomic programs by MoA state", fontweight="bold")
    fig.tight_layout()
    path = figs / "drugseq_program_state_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    summary_out = summary.reset_index().rename(columns={"state_class": "state"})
    return path, summary_out


def make_correlation_heatmap(correlations: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    pheno_cols = [col for col in ["permito", "mitomass", "area", "intensity"] if col in correlations]
    corr = correlations.set_index("pathway")[pheno_cols].copy()
    corr.index = [PROGRAM_LABELS.get(f"path_{idx}", idx) for idx in corr.index]

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-0.55, vmax=0.55, aspect="auto")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_xticks(np.arange(len(pheno_cols)))
    ax.set_xticklabels([PHENO_LABELS.get(col, col) for col in pheno_cols], rotation=25, ha="right")
    for y in range(corr.shape[0]):
        for x in range(corr.shape[1]):
            ax.text(x, y, f"{corr.iloc[y, x]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("target-level Spearman r")
    ax.set_title("DRUG-seq programs vs matched phenotype axes", fontweight="bold")
    fig.tight_layout()
    path = figs / "drugseq_program_phenotype_correlations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_candidate_bubble(target_table: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    data = target_table[target_table["category"].eq("Target")].copy()
    data = data[data["recommendation"].isin(["tier1_immediate_validation", "tier2_review"])].copy()
    if data.empty:
        data = target_table[target_table["category"].eq("Target")].copy().nlargest(35, "consensus_score")
    else:
        data = data.nlargest(45, "consensus_score")
    sizes = 48 + 18 * data["consensus_score"].fillna(0).clip(lower=0)

    fig, ax = plt.subplots(figsize=(11.4, 8.2))
    for state in safe_state_order(data["state_class"]):
        mask = data["state_class"].eq(state)
        ax.scatter(
            data.loc[mask, "permito"],
            data.loc[mask, "mitomass"],
            s=sizes.loc[mask],
            alpha=0.72,
            color=STATE_COLORS.get(state, "#666666"),
            label=STATE_LABELS.get(state, state),
            edgecolor="white",
            linewidth=0.7,
        )
    for _, row in data.nlargest(14, "consensus_score").iterrows():
        ax.text(row["permito"], row["mitomass"], str(row["group"]), fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#9aa0a6", lw=0.8)
    ax.axvline(0, color="#9aa0a6", lw=0.8)
    ax.set_xlabel("per-mito dPsi phenotype reference")
    ax.set_ylabel("mito mass phenotype reference")
    ax.set_title("High-priority DRUG-seq-supported candidates", fontweight="bold")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path = figs / "drugseq_candidate_bubble.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_toxicity_plot(target_table: pd.DataFrame, out: Path) -> tuple[Path, pd.DataFrame]:
    figs = out / "figs"
    data = target_table[target_table["category"].eq("Target")].copy()
    data["tox_rate"] = data["tox_rate"].fillna(0)
    xcol = "path_APOPTOSIS_TOXICITY"
    ycol = "path_PROTEOSTASIS_AUTOPHAGY"
    fig, ax = plt.subplots(figsize=(11.4, 8.0))
    for state in safe_state_order(data["state_class"]):
        mask = data["state_class"].eq(state)
        ax.scatter(
            data.loc[mask, xcol],
            data.loc[mask, ycol],
            s=48 + 220 * data.loc[mask, "tox_rate"],
            alpha=0.76,
            color=STATE_COLORS.get(state, "#666666"),
            label=STATE_LABELS.get(state, state),
            edgecolor="white",
            linewidth=0.6,
        )
    tox_targets = data[data["state_class"].eq("toxic_collapse")].sort_values("tox_rate", ascending=False)
    for _, row in tox_targets.iterrows():
        ax.text(row[xcol], row[ycol], str(row["group"]), fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#9aa0a6", lw=0.8)
    ax.axvline(0, color="#9aa0a6", lw=0.8)
    ax.set_xlabel("Apoptosis / toxicity DRUG-seq program")
    ax.set_ylabel("Proteostasis / autophagy DRUG-seq program")
    ax.set_title("Toxic-collapse transcriptional stress signature", fontweight="bold")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figs / "drugseq_toxicity_programs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, tox_targets


def summarize_state_tests(state_tests: pd.DataFrame) -> pd.DataFrame:
    if state_tests.empty:
        return pd.DataFrame()
    keep = state_tests.copy()
    keep = keep.sort_values(["fdr", "effect_mean_diff"], ascending=[True, False])
    cols = [
        "contrast",
        "feature_label",
        "n_case",
        "n_reference",
        "case_mean",
        "reference_mean",
        "effect_mean_diff",
        "fdr",
    ]
    return keep[[col for col in cols if col in keep.columns]].head(16)


def build_summary_tables(
    target_table: pd.DataFrame,
    expression_meta: pd.DataFrame,
    program_state_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    state_tests: pd.DataFrame,
    out: Path,
) -> dict[str, pd.DataFrame]:
    targets = target_table[target_table["category"].eq("Target")].copy()
    state_counts = (
        targets.groupby("state_class")
        .agg(
            n_targets=("group", "count"),
            median_consensus=("consensus_score", "median"),
            median_permito=("permito", "median"),
            median_mitomass=("mitomass", "median"),
            median_area=("area", "median"),
            mean_tox_rate=("tox_rate", "mean"),
        )
        .reindex(safe_state_order(targets["state_class"]))
        .reset_index()
    )
    state_counts["state_label"] = state_counts["state_class"].map(lambda value: STATE_LABELS.get(value, value))

    priority_order = {"tier1_immediate_validation": 0, "tier2_review": 1, "watchlist_or_context": 2}
    top_candidates = targets.copy()
    top_candidates["priority_order"] = top_candidates["recommendation"].map(priority_order).fillna(9)
    top_candidates = top_candidates.sort_values(
        ["priority_order", "consensus_score", "n_clean_wells"], ascending=[True, False, False]
    )
    top_candidates = top_candidates[[
        "group",
        "state_class",
        "recommendation",
        "kd_tier",
        "consensus_score",
        "permito",
        "mitomass",
        "area",
        "tox_rate",
        "path_OXPHOS_ETC",
        "path_MITO_BIOGENESIS",
        "path_APOPTOSIS_TOXICITY",
    ]].head(30)

    tox_targets = targets[targets["state_class"].eq("toxic_collapse")].copy()
    tox_targets = tox_targets.sort_values(["tox_rate", "path_APOPTOSIS_TOXICITY", "area"], ascending=[False, False, True])
    tox_targets = tox_targets[[
        "group",
        "kd_tier",
        "tox_rate",
        "permito",
        "area",
        "path_APOPTOSIS_TOXICITY",
        "path_ISR_ER_STRESS",
        "path_PROTEOSTASIS_AUTOPHAGY",
        "recommendation",
    ]]

    corr_long = correlations.melt(id_vars="pathway", var_name="axis", value_name="spearman")
    corr_long = corr_long[corr_long["axis"].isin(PHENO_LABELS)]
    corr_long["abs_spearman"] = corr_long["spearman"].abs()
    corr_long["pathway_label"] = corr_long["pathway"].map(lambda value: PROGRAM_LABELS.get(f"path_{value}", value))
    corr_long["axis_label"] = corr_long["axis"].map(lambda value: PHENO_LABELS.get(value, value))
    top_correlations = corr_long.sort_values("abs_spearman", ascending=False).head(16)

    key_state_tests = summarize_state_tests(state_tests)

    expression_meta_out = expression_meta[[
        "group",
        "state_class",
        "n_wells_expression",
        "kd_tier",
        "consensus_score",
        "drugseq_pc1",
        "drugseq_pc2",
    ]].sort_values(["state_class", "group"])

    tables = {
        "state_summary": state_counts,
        "top_candidates": top_candidates,
        "toxic_collapse_targets": tox_targets,
        "top_program_correlations": top_correlations,
        "key_state_tests": key_state_tests,
        "program_state_summary": program_state_summary,
        "expression_pca_coordinates": expression_meta_out,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


def render_html(
    out: Path,
    summary: dict,
    figures: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    target_table: pd.DataFrame,
) -> None:
    targets = target_table[target_table["category"].eq("Target")].copy()
    n_targets = int(targets["group"].nunique())
    n_tier1 = int(targets["recommendation"].eq("tier1_immediate_validation").sum())
    n_toxic = int(targets["state_class"].eq("toxic_collapse").sum())
    n_uncoupling = int(targets["state_class"].isin(["uncoupler_like", "mixed_uncoupling_biogenesis"]).sum())
    strong_or_weak = int(targets["kd_tier"].isin(["strong", "weak"]).sum())
    qc_fail_rate = summary.get("n_qc_fail", 0) / max(summary.get("n_wells", 1), 1)
    toxic_well_rate = summary.get("n_tox_flag", 0) / max(summary.get("n_wells", 1), 1)

    source_files = [
        "data/processed/adata_drugseq_processed.h5ad",
        "data/processed/prep_summary.json",
        "output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv",
        "output/2026-06-22/ba_multimodal_plan/pathway_phenotype_correlations.csv",
        "output/2026-06-22/ba_multimodal_plan/state_moa_tests.csv",
    ]
    source_html = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in source_files)

    fig = {name: figure_path(path, out) for name, path in figures.items()}

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EE DRUG-seq 专项分析报告</title>
  <style>
    :root {{
      --ink: #152126;
      --muted: #66737a;
      --line: #d9e0df;
      --paper: #f7f4ed;
      --panel: #ffffff;
      --teal: #176d6b;
      --green: #2d7d4f;
      --amber: #c9822a;
      --red: #b83c3c;
      --blue: #2c628f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f7f4ed 0%, #eef4f1 48%, #f9faf8 100%);
      color: var(--ink);
      font-family: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
      line-height: 1.62;
    }}
    header {{
      padding: 46px 42px 34px;
      background:
        radial-gradient(circle at 85% 12%, rgba(23, 109, 107, 0.18), transparent 31%),
        linear-gradient(135deg, #12343a 0%, #176d6b 58%, #d6a35b 100%);
      color: #ffffff;
    }}
    .wrap {{ max-width: 1380px; margin: 0 auto; padding: 0 28px 42px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px; opacity: 0.82; }}
    h1 {{ margin: 10px 0 10px; font-size: clamp(30px, 4vw, 52px); line-height: 1.08; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    h3 {{ margin: 20px 0 10px; font-size: 18px; }}
    .subtitle {{ max-width: 880px; font-size: 17px; opacity: 0.9; }}
    .section {{
      margin-top: 26px;
      padding: 26px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(35, 52, 55, 0.08);
    }}
    .grid {{ display: grid; gap: 16px; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .figure-stack {{ grid-template-columns: 1fr; gap: 24px; }}
    .metric {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfa;
    }}
    .metric .value {{ display: block; font-size: 28px; font-weight: 800; color: var(--teal); }}
    .metric .label {{ display: block; color: var(--muted); font-size: 13px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 7px; border: 1px solid var(--line); background: #fff; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.data th, table.data td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    table.data th {{ background: #edf3f0; color: #24363b; position: sticky; top: 0; }}
    .table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    .callout {{
      padding: 16px 18px;
      border-left: 5px solid var(--teal);
      background: #edf6f2;
      border-radius: 7px;
    }}
    .muted {{ color: var(--muted); }}
    .pill {{ display: inline-block; padding: 3px 9px; margin: 2px 4px 2px 0; border-radius: 999px; background: #e8efed; color: #294346; font-size: 12px; }}
    code {{ background: #eef2ef; padding: 2px 5px; border-radius: 4px; }}
    ul {{ padding-left: 20px; }}
    @media (max-width: 820px) {{
      header {{ padding: 34px 22px 26px; }}
      .wrap {{ padding: 0 14px 30px; }}
      .grid.two, .grid.three {{ grid-template-columns: 1fr; }}
      .section {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="eyebrow">EE Campaign | DRUG-seq 专项报告</div>
      <h1>从扰动转录组读出线粒体 MoA、命中和毒性风险</h1>
      <p class="subtitle">本报告只聚焦 DRUG-seq 数据层：样本 QC、KD 质量、target-level 表达签名、通路程序、MoA 状态和 toxic-collapse 读出。成像 phenotype 仅作为匹配参考轴，帮助解释转录组信号是否落在期望方向。</p>
    </div>
  </header>
  <main class="wrap">
    <section class="section">
      <h2>一句话结论</h2>
      <div class="callout">
        DRUG-seq 在这批 EE siRNA campaign 中已经足够作为机制判读层：它能把单纯 TMRM phenotype 拆成 uncoupling、biogenesis、mixed response 和 toxic collapse，并且为后续 virtual-cell benchmark 提供 target-level perturbation response ground truth。
      </div>
      <div class="grid three" style="margin-top: 18px;">
        <div class="metric"><span class="value">{summary.get('n_wells', '')}</span><span class="label">DRUG-seq wells</span></div>
        <div class="metric"><span class="value">{summary.get('n_genes', '')}</span><span class="label">genes measured</span></div>
        <div class="metric"><span class="value">{summary.get('n_hvg', '')}</span><span class="label">high-variable genes used</span></div>
        <div class="metric"><span class="value">{n_targets}</span><span class="label">target perturbations analyzed</span></div>
        <div class="metric"><span class="value">{strong_or_weak}</span><span class="label">targets with strong/weak KD</span></div>
        <div class="metric"><span class="value">{n_tier1}</span><span class="label">tier-1 validation candidates</span></div>
      </div>
    </section>

    <section class="section">
      <h2>1. 数据质量和扰动质量</h2>
      <p>总 QC-fail well 比例为 <b>{qc_fail_rate:.1%}</b>，tox-flag well 比例为 <b>{toxic_well_rate:.1%}</b>。plate-wise NTC 标准化后，批次结构明显减弱，说明后续 target-level signature 可以用于跨 plate 汇总。</p>
    <div class="grid figure-stack">
        <figure>
          <img src="{fig['qc_pca']}" alt="DRUG-seq QC PCA">
          <figcaption>同一批数据在 raw log-normalized HVG 和 within-NTC z-score HVG 下的 PCA。右图是后续分析使用的标准化空间。</figcaption>
        </figure>
        <figure>
          <img src="{fig['state_counts']}" alt="state counts and knockdown tiers">
          <figcaption>MoA 状态分布和 KD tier 分布。KD tier 用来判断 perturbation response 是否可以作为高置信机制证据。</figcaption>
        </figure>
      </div>
    </section>

    <section class="section">
      <h2>2. DRUG-seq 表达签名空间</h2>
      <p>每个 target 先在 clean wells 中取平均表达 z-score，形成 target-level DRUG-seq signature。这个空间不是简单的 hit ranking，而是机制 fingerprint：相邻 target 往往共享更相近的转录反应。</p>
      <figure>
        <img src="{fig['target_pca']}" alt="target-level DRUG-seq PCA">
        <figcaption>target-level DRUG-seq signature PCA。标注了最高 consensus 的候选和 toxic-collapse targets。</figcaption>
      </figure>
    </section>

    <section class="section">
      <h2>3. 通路程序：DRUG-seq 如何解释 MoA 状态</h2>
      <p>每个 MoA state 都有相对稳定的 transcriptomic program 均值。报告中的 program score 是 target-level signature 上的通路读出，适合做机制审稿而不是单独作为 hit gate。</p>
    <div class="grid figure-stack">
        <figure>
          <img src="{fig['program_state']}" alt="program state heatmap">
          <figcaption>按 MoA state 聚合的 DRUG-seq program score。toxic-collapse 的 apoptosis/toxicity 和 stress 读出最值得关注。</figcaption>
        </figure>
        <figure>
          <img src="{fig['program_pheno_corr']}" alt="program phenotype correlations">
          <figcaption>DRUG-seq program 与匹配 phenotype 轴的 target-level Spearman 相关。用于判断转录组和表型方向是否一致。</figcaption>
        </figure>
      </div>
      <h3>最强 program-phenotype 关联</h3>
      <div class="table-scroll">{html_table(tables['top_program_correlations'][['pathway_label', 'axis_label', 'spearman', 'abs_spearman']], max_rows=12)}</div>
    </section>

    <section class="section">
      <h2>4. 候选 target 和验证优先级</h2>
      <p>这里的优先级来自 DRUG-seq signature、matched phenotype 和状态一致性的综合读出。tier-1 候选适合进入下一轮 wet-lab validation；tier-2 更适合机制复核或与外部证据结合后再推进。</p>
      <figure>
        <img src="{fig['candidate_bubble']}" alt="candidate bubble plot">
        <figcaption>高优先级候选在 per-mito dPsi 和 mito mass 参考轴上的位置；点大小代表 consensus score。</figcaption>
      </figure>
      <h3>Top validation candidates</h3>
      <div class="table-scroll">{html_table(tables['top_candidates'], max_rows=18)}</div>
    </section>

    <section class="section">
      <h2>5. Toxic-collapse: DRUG-seq 毒性读出</h2>
      <p>本批严格 toxic-collapse target 为 <b>{n_toxic}</b> 个。它们不是“线粒体调节命中”的优先方向，而更像 assay 内部的毒性和 essentiality sentinel：细胞数下降、TMRM area collapse，同时伴随 apoptosis/toxicity、ISR/ER stress 或 proteostasis/autophagy 转录反应。</p>
      <figure>
        <img src="{fig['toxicity']}" alt="toxicity program plot">
        <figcaption>tox_rate 通过点大小体现；toxic-collapse target 被标注，展示其 stress/toxicity transcriptomic signature。</figcaption>
      </figure>
      <h3>Toxic-collapse targets</h3>
      <div class="table-scroll">{html_table(tables['toxic_collapse_targets'], max_rows=12)}</div>
    </section>

    <section class="section">
      <h2>6. 状态差异的统计证据</h2>
      <p>下面保留最显著的 state-vs-reference 检验，作为报告审计表。最强证据来自 uncoupling state 的 per-mito dPsi / TMRM area 下降、mixed state 的 mito mass 增加，以及 toxic-collapse 的 tox_rate 与 apoptosis/toxicity 上升。</p>
      <div class="table-scroll">{html_table(tables['key_state_tests'], max_rows=16)}</div>
    </section>

    <section class="section">
      <h2>7. 可复用输出</h2>
      <p>本报告同时输出了可复查 CSV 和图片，方便后续写 PPT、更新 Jira 或作为 benchmark 的 input manifest。</p>
      <p>
        <span class="pill">state_summary.csv</span>
        <span class="pill">top_candidates.csv</span>
        <span class="pill">toxic_collapse_targets.csv</span>
        <span class="pill">top_program_correlations.csv</span>
        <span class="pill">key_state_tests.csv</span>
        <span class="pill">expression_pca_coordinates.csv</span>
      </p>
      <h3>数据来源</h3>
      <ul>{source_html}</ul>
    </section>
  </main>
</body>
</html>
"""
    (out / "EE_DRUG_seq_dedicated_analysis_report.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    import anndata as ad

    processed = Path(args.processed)
    adata = ad.read_h5ad(args.adata)
    with open(processed / "prep_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    target_table = pd.read_csv(args.target_table)
    correlations = pd.read_csv(args.correlations)
    state_tests = pd.read_csv(args.state_tests)

    qc_pca = make_qc_pca(adata, out)
    target_pca, expression_meta = make_target_pca(adata, target_table, out, args.min_target_wells)
    state_counts = make_state_count_plot(target_table, out)
    program_state, program_state_summary = make_program_state_heatmap(target_table, out)
    program_pheno_corr = make_correlation_heatmap(correlations, out)
    candidate_bubble = make_candidate_bubble(target_table, out)
    toxicity, _ = make_toxicity_plot(target_table, out)

    tables = build_summary_tables(
        target_table,
        expression_meta,
        program_state_summary,
        correlations,
        state_tests,
        out,
    )
    figures = {
        "qc_pca": qc_pca,
        "target_pca": target_pca,
        "state_counts": state_counts,
        "program_state": program_state,
        "program_pheno_corr": program_pheno_corr,
        "candidate_bubble": candidate_bubble,
        "toxicity": toxicity,
    }
    render_html(out, summary, figures, tables, target_table)

    manifest = {
        "report": "EE_DRUG_seq_dedicated_analysis_report.html",
        "figures": {name: path.relative_to(out).as_posix() for name, path in figures.items()},
        "tables": [f"{name}.csv" for name in tables],
        "sources": {
            "adata": args.adata,
            "target_table": args.target_table,
            "correlations": args.correlations,
            "state_tests": args.state_tests,
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[drugseq_report] wrote {out / 'EE_DRUG_seq_dedicated_analysis_report.html'}")
    print(f"[drugseq_report] figures: {len(figures)} | tables: {len(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())