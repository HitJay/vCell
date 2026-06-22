#!/usr/bin/env python
"""Cross-modal MoA analysis for the B+A route.

This script shifts the B+A route from target prioritization to mechanism
understanding. It combines three target-level views:

* transcriptomic reference connectivity on plate-wise NTC-z HVG signatures;
* curated pathway scores computed from log-normalized expression and converted
  to plate-wise NTC z-scores;
* phenotype/imaging state variables from the B+A state atlas and OOF multimodal
  baseline predictions.

Outputs are designed as figure-ready MoA evidence rather than a validation
shortlist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr, ttest_ind  # noqa: E402


REFERENCE_GROUPS = ["BAM15", "MK8722", "PSMC3", "ATP5B", "SLC25A4"]

PATHWAYS: dict[str, list[str]] = {
    "OXPHOS_ETC": [
        "NDUFA1", "NDUFA9", "NDUFB8", "NDUFS1", "NDUFS2", "NDUFV1",
        "SDHA", "SDHB", "UQCRC1", "UQCRC2", "CYC1", "COX4I1", "COX5A",
        "COX6A1", "COX7A2", "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5MC1",
        "MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND4",
    ],
    "MITO_BIOGENESIS": [
        "PPARGC1A", "PPARGC1B", "NRF1", "GABPA", "TFAM", "TFB1M", "TFB2M",
        "POLG", "POLG2", "TWNK", "SIRT1", "SIRT3", "SIRT4", "ESRRA",
    ],
    "FAO_LIPID": [
        "CPT1A", "CPT2", "ACADM", "ACADVL", "ACADS", "HADHA", "HADHB",
        "ECHS1", "ACOX1", "PPARA", "PPARD", "SLC25A20", "DGAT1", "DGAT2",
        "FADS1", "FADS2", "SCD", "CD36",
    ],
    "AMPK_MTOR_INSULIN": [
        "PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "STK11", "CAMKK2",
        "MTOR", "RPTOR", "RICTOR", "TSC1", "TSC2", "AKT1", "AKT2", "IRS1",
        "INSR", "MLX", "MLXIPL", "GCKR",
    ],
    "ISR_ER_STRESS": [
        "ATF4", "ATF5", "DDIT3", "EIF2AK3", "ERN1", "ATF6", "XBP1", "HSPA5",
        "HYOU1", "DNAJB9", "HERPUD1", "PPP1R15A", "TRIB3", "ASNS", "CHAC1",
    ],
    "PROTEOSTASIS_AUTOPHAGY": [
        "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMD1", "PSMD2", "SQSTM1",
        "MAP1LC3B", "ATG5", "ATG7", "BECN1", "ULK1", "PINK1", "PRKN", "OPTN",
    ],
    "APOPTOSIS_TOXICITY": [
        "BAX", "BAK1", "BCL2", "MCL1", "BID", "CASP3", "CASP7", "CASP8",
        "CASP9", "PMAIP1", "BBC3", "GADD45A", "CDKN1A", "HMOX1", "NQO1",
    ],
}

PHENO_COLS = ["permito", "mitomass", "area", "intensity"]
REF_COLS = [f"conn_{ref}" for ref in REFERENCE_GROUPS]


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def dense_slice(matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    sub = matrix[rows][:, cols]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


def platewise_ntc_z(values: np.ndarray, plate: np.ndarray, is_ntc: np.ndarray, clip: float = 10.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    z = np.full_like(values, np.nan, dtype=np.float64)
    for pl in pd.unique(plate):
        mask = plate == pl
        ref = mask & is_ntc
        if ref.sum() < 2:
            ref = mask
        mu = np.nanmean(values[ref], axis=0)
        sd = np.nanstd(values[ref], axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        z[mask] = (values[mask] - mu) / sd
    return np.clip(z, -clip, clip)


def corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    if np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0] if method == "spearman" else pearsonr(a[ok], b[ok])[0])


def load_clean_obs(adata) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["plate"] = obs["plate"].astype(str)
    obs["category"] = obs["category"].astype(str)
    obs["qc_fail"] = as_bool(obs["qc_fail"])
    obs["tox_flag"] = as_bool(obs["tox_flag"])
    return obs


def target_mean(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return df.groupby("group", as_index=False)[value_cols].mean(numeric_only=True)


def build_signature_table(adata, obs: pd.DataFrame, clean_mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    hvg_symbols = adata.var.loc[hvg_mask, "symbol"].astype(str).tolist()
    X = np.asarray(adata.obsm["X_zscore_hvg"], dtype=np.float32)
    clean_obs = obs.loc[clean_mask].reset_index(drop=True)
    clean_X = X[clean_mask]
    rows = []
    vectors = []
    for group, idx in clean_obs.groupby("group").indices.items():
        vec = clean_X[np.asarray(idx)].mean(axis=0)
        rows.append({"group": group, "n_signature_wells": len(idx)})
        vectors.append(vec)
    sig = pd.DataFrame(rows)
    sig_matrix = np.vstack(vectors) if vectors else np.empty((0, clean_X.shape[1]))
    return sig, sig_matrix, hvg_symbols


def add_reference_connectivity(sig: pd.DataFrame, sig_matrix: np.ndarray, method: str) -> pd.DataFrame:
    out = sig.copy()
    group_to_idx = {g: i for i, g in enumerate(out["group"])}
    for ref in REFERENCE_GROUPS:
        col = f"conn_{ref}"
        if ref not in group_to_idx:
            out[col] = np.nan
            continue
        ref_vec = sig_matrix[group_to_idx[ref]]
        out[col] = [corr(sig_matrix[i], ref_vec, method=method) for i in range(sig_matrix.shape[0])]
    out["conn_BAM15_minus_MK8722"] = out["conn_BAM15"] - out["conn_MK8722"]
    out["conn_MK8722_minus_BAM15"] = out["conn_MK8722"] - out["conn_BAM15"]
    out["conn_toxicity_margin"] = out["conn_PSMC3"] - out[["conn_BAM15", "conn_MK8722"]].max(axis=1)
    return out


def build_pathway_scores(adata, obs: pd.DataFrame) -> pd.DataFrame:
    symbols = adata.var["symbol"].astype(str).to_numpy()
    symbol_to_idx: dict[str, int] = {}
    for i, symbol in enumerate(symbols):
        symbol_to_idx.setdefault(symbol, i)

    score_df = pd.DataFrame({"group": obs["group"].to_numpy(), "plate": obs["plate"].to_numpy(), "category": obs["category"].to_numpy()})
    is_ntc = obs["group"].eq("NTC").to_numpy()
    plate = obs["plate"].to_numpy()
    for name, genes in PATHWAYS.items():
        idx = np.asarray([symbol_to_idx[g] for g in genes if g in symbol_to_idx], dtype=int)
        if len(idx) == 0:
            score_df[f"path_{name}"] = np.nan
            score_df[f"path_{name}_n_genes"] = 0
            continue
        vals = dense_slice(adata.layers["lognorm"], np.arange(adata.n_obs), idx).mean(axis=1)
        z = platewise_ntc_z(vals[:, None], plate, is_ntc)[:, 0]
        score_df[f"path_{name}"] = z
        score_df[f"path_{name}_n_genes"] = len(idx)
    path_cols = [c for c in score_df.columns if c.startswith("path_") and not c.endswith("_n_genes")]
    clean_mask = (obs["category"].isin(["Target", "PC"]) & (~obs["qc_fail"]) & (~obs["tox_flag"])).to_numpy()
    clean = score_df.loc[clean_mask]
    return clean.groupby("group", as_index=False)[path_cols].mean(numeric_only=True)


def load_model_predictions(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame({"group": []})
    pred = pd.read_csv(p)
    pred = pred[(pred["scheme"] == "logo_target") & (pred["model"] == "ridge")]
    keep_features = ["expression", "expression_c1", "c24_mito", "expression_c1_c24"]
    pred = pred[pred["feature_set"].isin(keep_features)]
    rows = []
    for (group, outcome), sub in pred.groupby(["group", "outcome"]):
        rec = {"group": group, "outcome": outcome}
        rec["y_true"] = sub["y_true"].mean()
        for _, row in sub.iterrows():
            rec[f"pred_{row['feature_set']}"] = row["y_pred"]
        rows.append(rec)
    wide = pd.DataFrame(rows)
    if wide.empty:
        return pd.DataFrame({"group": []})
    return wide.pivot(index="group", columns="outcome").reset_index().pipe(flatten_columns)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if part]
            cols.append("_".join(parts))
        else:
            cols.append(str(col))
    df.columns = cols
    if "group_" in df.columns:
        df = df.rename(columns={"group_": "group"})
    return df


def build_target_table(args: argparse.Namespace):
    import anndata as ad

    adata = ad.read_h5ad(args.adata)
    obs = load_clean_obs(adata)
    clean_mask = ((~obs["qc_fail"]) & (~obs["tox_flag"]) & obs["category"].isin(["Target", "PC", "NC"])).to_numpy()

    sig, sig_matrix, hvg_symbols = build_signature_table(adata, obs, clean_mask)
    conn = add_reference_connectivity(sig, sig_matrix, method=args.connectivity)
    path_scores = build_pathway_scores(adata, obs)
    atlas = pd.read_csv(args.atlas)
    model_pred = load_model_predictions(args.oof_predictions)

    table = atlas.merge(conn, on="group", how="left")
    table = table.merge(path_scores, on="group", how="left")
    if not model_pred.empty:
        table = table.merge(model_pred, on="group", how="left")
    return table, sig, sig_matrix, hvg_symbols


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full_like(pvalues, np.nan, dtype=float)
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


def compute_state_volcano(
    sig: pd.DataFrame,
    sig_matrix: np.ndarray,
    hvg_symbols: list[str],
    table: pd.DataFrame,
) -> pd.DataFrame:
    meta = sig[["group"]].merge(table[["group", "category", "state_class"]], on="group", how="left")
    contrasts = {
        "uncoupling_states_vs_neutral": ["uncoupler_like", "mixed_uncoupling_biogenesis"],
        "biogenesis_like_vs_neutral": ["biogenesis_like"],
        "toxic_collapse_vs_neutral": ["toxic_collapse"],
    }
    rows = []
    reference = (meta["category"].eq("Target") & meta["state_class"].eq("neutral_or_uncertain")).to_numpy()
    for contrast, states in contrasts.items():
        case = (meta["category"].eq("Target") & meta["state_class"].isin(states)).to_numpy()
        if case.sum() < 3 or reference.sum() < 3:
            continue
        case_x = sig_matrix[case]
        ref_x = sig_matrix[reference]
        effect = case_x.mean(axis=0) - ref_x.mean(axis=0)
        stat, pvalue = ttest_ind(case_x, ref_x, axis=0, equal_var=False, nan_policy="omit")
        qvalue = fdr_bh(pvalue)
        for i, symbol in enumerate(hvg_symbols):
            rows.append(
                {
                    "contrast": contrast,
                    "symbol": symbol,
                    "effect_ntc_z": float(effect[i]),
                    "t_stat": float(stat[i]) if np.isfinite(stat[i]) else np.nan,
                    "p_value": float(pvalue[i]) if np.isfinite(pvalue[i]) else np.nan,
                    "fdr": float(qvalue[i]) if np.isfinite(qvalue[i]) else np.nan,
                    "n_case_targets": int(case.sum()),
                    "n_reference_targets": int(reference.sum()),
                }
            )
    volcano = pd.DataFrame(rows)
    if volcano.empty:
        return volcano
    volcano["neg_log10_fdr"] = -np.log10(volcano["fdr"].clip(lower=1e-300))
    pathway_genes = {gene for genes in PATHWAYS.values() for gene in genes}
    volcano["curated_pathway_gene"] = volcano["symbol"].isin(pathway_genes)
    volcano["significant"] = (volcano["fdr"] < 0.1) & (volcano["effect_ntc_z"].abs() >= 1.0)
    return volcano


def plot_state_volcano(volcano: pd.DataFrame, figs: Path) -> None:
    if volcano.empty:
        return
    contrasts = list(volcano["contrast"].unique())
    fig, axes = plt.subplots(1, len(contrasts), figsize=(5.3 * len(contrasts), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, contrast in zip(axes, contrasts):
        sub = volcano[volcano["contrast"].eq(contrast)].copy()
        colors = np.where(
            sub["significant"] & (sub["effect_ntc_z"] > 0),
            "#b2182b",
            np.where(sub["significant"] & (sub["effect_ntc_z"] < 0), "#2166ac", "#bdbdbd"),
        )
        ax.scatter(sub["effect_ntc_z"], sub["neg_log10_fdr"], s=10, c=colors, alpha=0.72, linewidth=0)
        pathway = sub[sub["curated_pathway_gene"]]
        ax.scatter(pathway["effect_ntc_z"], pathway["neg_log10_fdr"], s=24, facecolors="none", edgecolors="#fdae61", linewidth=0.7)
        ax.axvline(0, color="grey", lw=0.7)
        ax.axvline(1, color="grey", lw=0.5, ls="--")
        ax.axvline(-1, color="grey", lw=0.5, ls="--")
        ax.axhline(-np.log10(0.1), color="grey", lw=0.5, ls="--")
        labels = sub.assign(label_score=sub["effect_ntc_z"].abs() * sub["neg_log10_fdr"])
        labels = labels.sort_values("label_score", ascending=False).head(8)
        for _, row in labels.iterrows():
            ax.annotate(row["symbol"], (row["effect_ntc_z"], row["neg_log10_fdr"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        title = contrast.replace("_", " ")
        ax.set_title(title)
        ax.set_xlabel("mean state effect vs neutral (NTC-z expression)")
    axes[0].set_ylabel("-log10(FDR)")
    fig.suptitle("Transcriptomic state volcano plots (target-level HVG signatures)")
    fig.tight_layout()
    fig.savefig(figs / "transcriptomic_state_volcano.png", dpi=150)
    plt.close(fig)


def zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    return (out - out.mean(axis=0)) / out.std(axis=0).replace(0, np.nan)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return PCA(n_components=2, random_state=0).fit_transform(matrix)


def plot_moa_map(table: pd.DataFrame, figs: Path) -> None:
    path_cols = [c for c in table.columns if c.startswith("path_")]
    feature_cols = PHENO_COLS + REF_COLS + path_cols
    plot_df = table[table["category"].isin(["Target", "PC"])].copy()
    X = zscore_cols(plot_df, feature_cols).to_numpy()
    emb = pca_2d(X)
    plot_df["moa_pc1"] = emb[:, 0]
    plot_df["moa_pc2"] = emb[:, 1]

    color_map = {
        "uncoupler_like": "#d95f02",
        "mixed_uncoupling_biogenesis": "#7570b3",
        "biogenesis_like": "#1b9e77",
        "energizer_like": "#66a61e",
        "toxic_collapse": "#b2182b",
        "neutral_or_uncertain": "#8c8c8c",
    }
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for state, sub in plot_df.groupby("state_class"):
        ax.scatter(sub["moa_pc1"], sub["moa_pc2"], s=45, alpha=0.78, c=color_map.get(state, "grey"), label=state)
    for group in REFERENCE_GROUPS:
        sub = plot_df[plot_df["group"] == group]
        if not sub.empty:
            row = sub.iloc[0]
            ax.scatter(row["moa_pc1"], row["moa_pc2"], marker="*", s=220, c="black", edgecolor="white", linewidth=0.7)
            ax.annotate(group, (row["moa_pc1"], row["moa_pc2"]), xytext=(5, 5), textcoords="offset points", fontsize=9, weight="bold")
    top = plot_df[plot_df["recommendation"].isin(["tier1_immediate_validation", "tier2_secondary_review"])].head(12)
    for _, row in top.iterrows():
        ax.annotate(row["group"], (row["moa_pc1"], row["moa_pc2"]), xytext=(4, -7), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Cross-modal MoA PC1")
    ax.set_ylabel("Cross-modal MoA PC2")
    ax.set_title("Cross-modal MoA map: phenotype + reference connectivity + pathway programs")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "crossmodal_moa_map.png", dpi=150)
    plt.close(fig)


def heatmap(data: pd.DataFrame, out: Path, title: str, figsize: tuple[float, float], cmap: str = "RdBu_r", vmin: float = -1, vmax: float = 1) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reference_heatmap(table: pd.DataFrame, figs: Path) -> None:
    candidates = table[table["category"].eq("Target")].copy()
    candidates = candidates.sort_values("phenotype_strength", ascending=False).head(40)
    hm = candidates.set_index("group")[REF_COLS]
    heatmap(hm, figs / "reference_connectivity_heatmap.png", "Transcriptomic reference connectivity (HVG signatures)", (7.5, 10))


def plot_pathway_correlations(table: pd.DataFrame, figs: Path) -> pd.DataFrame:
    path_cols = [c for c in table.columns if c.startswith("path_")]
    signal_cols = PHENO_COLS + REF_COLS + [
        c for c in table.columns
        if c.startswith("pred_expression_c1_") or c.startswith("pred_c24_mito_")
    ]
    target = table[table["category"].eq("Target")]
    rows = []
    for pathway in path_cols:
        rec = {"pathway": pathway.replace("path_", "")}
        for signal in signal_cols:
            if signal in target:
                rec[signal] = corr(target[pathway].to_numpy(), target[signal].to_numpy())
        rows.append(rec)
    corr_df = pd.DataFrame(rows).set_index("pathway")
    heatmap(corr_df, figs / "pathway_phenotype_correlation_heatmap.png", "Pathway programs vs cross-modal signals", (12, 6))
    return corr_df


def plot_state_summary(table: pd.DataFrame, figs: Path) -> pd.DataFrame:
    path_cols = [c for c in table.columns if c.startswith("path_")]
    cols = PHENO_COLS + REF_COLS + path_cols
    target = table[table["category"].eq("Target")].copy()
    state = target.groupby("state_class")[cols].mean(numeric_only=True)
    state_z = (state - state.mean(axis=0)) / state.std(axis=0).replace(0, np.nan)
    heatmap(state_z, figs / "state_moa_summary_heatmap.png", "State-level MoA summary (column z-score)", (13, 4.8))
    return state


def write_summary(table: pd.DataFrame, state_summary: pd.DataFrame, corr_df: pd.DataFrame, out: Path) -> None:
    target = table[table["category"].eq("Target")].copy()
    refs = target[REF_COLS + ["state_class"]].groupby("state_class").mean(numeric_only=True)
    strongest = []
    for signal in ["permito", "mitomass", "area", "conn_BAM15", "conn_MK8722", "conn_PSMC3"]:
        if signal in target:
            top = target.reindex(target[signal].abs().sort_values(ascending=False).index).head(8)
            strongest.append(f"### strongest {signal}\n" + top[["group", "state_class", signal, "tox_rate", "kd_tier"]].round(3).to_markdown(index=False))
    md = f"""# B+A Cross-modal MoA Report

This analysis is mechanism-oriented rather than prioritization-oriented. It asks whether transcriptomic programs, reference signatures, imaging-derived phenotypes, and model-predicted modalities organize targets into interpretable MoA states.

## Outputs

- `crossmodal_moa_target_table.csv` - target-level phenotype, reference connectivity, pathway scores and OOF prediction summaries.
- `crossmodal_moa_state_summary.csv` - state-level averages of MoA features.
- `pathway_phenotype_correlations.csv` - pathway-vs-signal correlation matrix.
- `figs/crossmodal_moa_map.png` - integrated MoA PCA map.
- `figs/reference_connectivity_heatmap.png` - target transcriptomic similarity to BAM15/MK8722/PSMC3/ATP5B/SLC25A4.
- `figs/pathway_phenotype_correlation_heatmap.png` - pathway programs aligned to phenotype/reference/model signals.
- `figs/state_moa_summary_heatmap.png` - state-level MoA summary.
- `figs/transcriptomic_state_volcano.png` - state-vs-neutral target-level transcriptomic volcano plots.
- `transcriptomic_state_volcano.csv` - gene-level volcano statistics for each state contrast.

## State-level reference connectivity

{refs.round(3).to_markdown()}

## State-level MoA means

{state_summary.round(3).to_markdown()}

## Pathway/signal correlations

{corr_df.round(3).to_markdown()}

## Strongest axes

{chr(10).join(strongest)}

## Working hypotheses for figures

1. If uncoupler-like and biogenesis-like states are real MoA states, they should separate in the integrated MoA map and show different reference/pathway correlation patterns.
2. If PSMC3-like toxicity is a distinct confounder, toxic-collapse targets should show higher PSMC3 connectivity and/or apoptosis/proteostasis programs.
3. If C24 is an upper-bound readout rather than a general virtual-cell predictor, C24 model predictions should align tightly with phenotype axes, while expression/BF signals should align more selectively with pathway programs.
"""
    (out / "B_A_crossmodal_moa_report.md").write_text(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--atlas", default="output/2026-06-22/ba_multimodal_plan/B_A_state_atlas.csv")
    parser.add_argument("--oof-predictions", default="output/2026-06-22/ba_multimodal_plan/multimodal_oof_target_predictions.csv")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    parser.add_argument("--connectivity", choices=["pearson", "spearman"], default="pearson")
    args = parser.parse_args(argv)

    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    table, sig, sig_matrix, hvg_symbols = build_target_table(args)
    state_summary = plot_state_summary(table, figs)
    corr_df = plot_pathway_correlations(table, figs)
    plot_moa_map(table, figs)
    plot_reference_heatmap(table, figs)
    volcano = compute_state_volcano(sig, sig_matrix, hvg_symbols, table)
    plot_state_volcano(volcano, figs)

    table.to_csv(out / "crossmodal_moa_target_table.csv", index=False)
    state_summary.to_csv(out / "crossmodal_moa_state_summary.csv")
    corr_df.to_csv(out / "pathway_phenotype_correlations.csv")
    volcano.to_csv(out / "transcriptomic_state_volcano.csv", index=False)
    write_summary(table, state_summary, corr_df, out)

    print(f"[ba_crossmodal_moa] wrote {out / 'crossmodal_moa_target_table.csv'}")
    print(f"[ba_crossmodal_moa] wrote {out / 'crossmodal_moa_state_summary.csv'}")
    print(f"[ba_crossmodal_moa] wrote {out / 'pathway_phenotype_correlations.csv'}")
    print(f"[ba_crossmodal_moa] wrote {out / 'B_A_crossmodal_moa_report.md'}")
    print(f"[ba_crossmodal_moa] figures under {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())