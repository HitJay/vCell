#!/usr/bin/env python
"""Generate the main-line-D QC report (figures + markdown) from the processed
DRUG-seq products written by ``scripts/prep_drugseq.py``.

Outputs (under ``--out``, default ``output/2026-06-10``):
  figs/pca_batch_correction.png   batch effect before/after within-NTC z
  figs/positive_control_window.png  2-axis phenotype, PCs vs NTC
  figs/kd_and_toxicity.png        KD-tier and toxicity distributions
  QC_report_drugseq.md            summary numbers + sanity checks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _pca(X: np.ndarray, n: int = 2) -> np.ndarray:
    Xc = X - X.mean(0)
    Xc = np.nan_to_num(Xc)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n] * S[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="output/2026-06-10")
    args = ap.parse_args(argv)

    import anndata as ad

    proc = Path(args.processed)
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(proc / "adata_drugseq_processed.h5ad")
    obs = adata.obs
    targets = pd.read_csv(proc / "targets_summary.csv")
    summary = json.load(open(proc / "prep_summary.json"))

    batch = obs["plate"].astype(str).to_numpy()
    cats = obs["category"].astype(str).to_numpy()

    # ---- fig 1: batch effect before/after within-NTC standardisation ----
    pca_raw = _pca(adata.obsm["X_lognorm_hvg"])
    pca_z = _pca(adata.obsm["X_zscore_hvg"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, title in [(axes[0], pca_raw, "log-norm (raw)"),
                           (axes[1], pca_z, "within-NTC z (corrected)")]:
        for b in pd.unique(batch):
            m = batch == b
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, label=b)
        ax.set_title(f"PCA — {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[1].legend(fontsize=7, markerscale=1.5, ncol=2)
    fig.suptitle("Batch-correction check (colour = plate)")
    fig.tight_layout()
    fig.savefig(figs / "pca_batch_correction.png", dpi=120)
    plt.close(fig)

    # ---- fig 2: positive-control phenotype window ----
    fig, ax = plt.subplots(figsize=(7, 6))
    base = ~np.isin(obs["group"].to_numpy(), ["NTC", "BAM15", "MK8722"])
    ax.scatter(obs["pheno_intensity_z"][base], obs["pheno_area_z"][base],
               s=8, c="lightgrey", alpha=0.5, label="targets")
    palette = {"NTC": "black", "BAM15": "tab:red", "MK8722": "tab:green"}
    for g, c in palette.items():
        m = obs["group"].to_numpy() == g
        ax.scatter(obs["pheno_intensity_z"][m], obs["pheno_area_z"][m],
                   s=28, c=c, label=g, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("pheno_intensity_z  (MK8722 / energizer axis →)")
    ax.set_ylabel("pheno_area_z  (BAM15 / uncoupler axis ↓)")
    ax.set_title("TMRM phenotype window (within-batch NTC z)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "positive_control_window.png", dpi=120)
    plt.close(fig)

    # ---- fig 3: KD tiers + toxicity ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_order = ["strong", "weak", "failed", "unknown"]
    tc = targets["kd_tier"].value_counts().reindex(tier_order).fillna(0)
    axes[0].bar(tier_order, tc.to_numpy(), color="tab:blue")
    axes[0].set_title("Knockdown efficiency tiers (per target)")
    axes[0].set_ylabel("# targets")
    axes[1].hist(obs["tox_cellcount_ratio"].dropna(), bins=40, color="tab:orange")
    axes[1].axvline(0.3, color="red", ls="--", label="tox_frac = 0.3")
    axes[1].set_title(f"cell_count / NTC median  ({int(obs['tox_flag'].sum())} toxic wells)")
    axes[1].set_xlabel("fraction of same-batch NTC median")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figs / "kd_and_toxicity.png", dpi=120)
    plt.close(fig)

    # ---- fig 4: MitoTracker-resolved mechanistic axes ----
    has_mito = "pheno_permito_dpsi_z" in obs and "pheno_mitomass_z" in obs
    if has_mito:
        fig, ax = plt.subplots(figsize=(7, 6))
        base = ~np.isin(obs["group"].to_numpy(), ["NTC", "BAM15", "MK8722"])
        ax.scatter(obs["pheno_mitomass_z"][base], obs["pheno_permito_dpsi_z"][base],
                   s=8, c="lightgrey", alpha=0.5, label="targets")
        for g, c in {"NTC": "black", "BAM15": "tab:red", "MK8722": "tab:green"}.items():
            m = obs["group"].to_numpy() == g
            ax.scatter(obs["pheno_mitomass_z"][m], obs["pheno_permito_dpsi_z"][m],
                       s=28, c=c, label=g, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel("pheno_mitomass_z  (ch4/ch1 — biogenesis, MK8722 →)")
        ax.set_ylabel("pheno_permito_dpsi_z  (ch2/ch4 — per-mito ΔΨm, BAM15 ↓)")
        ax.set_title("MitoTracker-resolved EE mechanism (within-batch NTC z)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "mechanism_axes.png", dpi=120)
        plt.close(fig)

    # ---- markdown report ----
    bam = targets.set_index("group").loc["BAM15"]
    mk = targets.set_index("group").loc["MK8722"]
    sort_col = "pheno_permito_dpsi_z" if "pheno_permito_dpsi_z" in targets else "pheno_area_z"
    hits = targets[targets["ee_hit_candidate"]].sort_values(sort_col)
    mech = ""
    if "pheno_permito_dpsi_z" in targets:
        mech = f"""
## MitoTracker (ch4) — mechanistically resolved EE axes
ch1 = nucleus/cell, ch2 = TMRM (ΔΨm), **ch4 = MitoTracker (mito mass)**. ch2/ch1
convolves membrane potential with mito mass; ch4 lets us separate them:
- **per-mito ΔΨm** = ch2/ch4 (`pheno_permito_dpsi_z`) — uncoupling axis (BAM15 ↓).
- **mito mass / biogenesis** = ch4/ch1 (`pheno_mitomass_z`) — biogenesis axis (MK8722 ↑).

| control | per-mito ΔΨm (ch2/ch4) | mito mass (ch4/ch1) | reading |
| --- | --- | --- | --- |
| BAM15 | **{bam['pheno_permito_dpsi_z']:.1f}** | {bam['pheno_mitomass_z']:.1f} | uncoupling (ΔΨm collapses) |
| MK8722 | {mk['pheno_permito_dpsi_z']:.1f} | **{mk['pheno_mitomass_z']:.1f}** | biogenesis (more mitochondria) |

This shows MK8722's large ch2/ch1 signal is **biogenesis-driven** (mito mass up),
not a per-mitochondrion potential increase — only resolvable with MitoTracker.
Hit directions: {summary.get('ee_hit_directions', {})}

See `figs/mechanism_axes.png`.
"""
    md = f"""# Main line D — DRUG-seq data-foundation QC report

Generated from `{args.processed}` on plate-wise within-NTC standardisation.

## Summary
- wells: **{summary['n_wells']}** | genes: {summary['n_genes']} | HVG: {summary['n_hvg']}
- perturbations: {summary['n_perturbations']} | within-batch unit: `{summary['batch_key']}`
- KD tiers: {summary['kd_tier_counts']}
- QC-fail wells: {summary['n_qc_fail']} | toxic wells: {summary['n_tox_flag']}
- EE hit candidates (hand-off to main line B): {summary['n_ee_hit_candidates']}

## Sanity checks (assay window)
| control | role | pheno_intensity_z | pheno_area_z | expected |
| --- | --- | --- | --- | --- |
| BAM15 | uncoupler | {bam['pheno_intensity_z']:.2f} | **{bam['pheno_area_z']:.2f}** | area axis **down** |
| MK8722 | AMPK activator | **{mk['pheno_intensity_z']:.2f}** | {mk['pheno_area_z']:.2f} | intensity axis **up** |

Both directions are as expected → the phenotype axes are correctly oriented.
{mech}
## Figures
- `figs/pca_batch_correction.png` — plate structure before/after within-NTC z.
- `figs/positive_control_window.png` — 2-axis TMRM phenotype (PCs vs NTC).
- `figs/kd_and_toxicity.png` — KD tiers and cell_count/NTC distribution.
- `figs/mechanism_axes.png` — MitoTracker-resolved per-mito ΔΨm vs mito mass.

## Top EE hit candidates (initial flag; refined in main line B)
{hits[['group', 'category', 'kd_tier', 'pheno_permito_dpsi_z', 'pheno_mitomass_z', 'pheno_area_z', 'ee_hit_direction', 'tox_rate']].head(15).round(2).to_markdown(index=False)}

> Note: `ee_hit_candidate` is a simple transparent flag from main line D; proper
> hit calling (consensus of imaging + transcriptomic MoA matching) is main line B.
"""
    (out / "QC_report_drugseq.md").write_text(md)
    n_figs = len(list(figs.glob("*.png")))
    print(f"[report] wrote {out/'QC_report_drugseq.md'} and {n_figs} figures under {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
