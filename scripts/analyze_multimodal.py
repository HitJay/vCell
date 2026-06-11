#!/usr/bin/env python
"""Tri-modal consistency analysis for the aligned HepG2 EE DRUG-seq object.

Quantifies, at the *target* level (batch-wise NTC-standardised, toxic wells
removed), how much the three modalities agree:

* TMRM mechanistic axes (our main-line-D phenotypes)
* vAssay Seahorse predictions (pred_MB / pred_AUC, pipeline TabPFN on C24)
* transcriptome (OXPHOS expression score)

It also reports per-modality reliability (split-half) so cross-modal
correlations can be read against each modality's own signal ceiling.

    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/analyze_multimodal.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

OXPHOS = ["NDUFB8", "SDHB", "UQCRC2", "COX5A", "ATP5F1B", "NDUFA1", "SDHA",
          "MT-CO1", "COX4I1", "UQCRC1", "NDUFS1", "CYC1"]


def batch_ntc_z(values: np.ndarray, plate: np.ndarray, is_ntc: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    z = np.full_like(v, np.nan)
    for b in pd.unique(plate):
        r = plate == b
        ntc = r & is_ntc
        ref = v[ntc] if ntc.sum() >= 2 else v[r]
        mu, sd = np.nanmean(ref), np.nanstd(ref)
        sd = sd if sd > 1e-9 else 1.0
        z[r] = (v[r] - mu) / sd
    return z


def split_half(df: pd.DataFrame, col: str, seed: int = 0) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    h1, h2 = {}, {}
    for g, sub in df.groupby("group"):
        if len(sub) < 4:
            continue
        idx = rng.permutation(len(sub))
        half = len(sub) // 2
        h1[g] = sub.iloc[idx[:half]][col].mean()
        h2[g] = sub.iloc[idx[half:]][col].mean()
    keys = [k for k in h1 if k in h2]
    if len(keys) < 4:
        return float("nan"), 0
    return pearsonr([h1[k] for k in keys], [h2[k] for k in keys])[0], len(keys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mm", default="data/processed/adata_multimodal.h5ad")
    ap.add_argument("--out", default="output/2026-06-10")
    args = ap.parse_args(argv)

    import anndata as ad

    a = ad.read_h5ad(args.mm)
    o = a.obs.copy()
    o["plate"] = o["plate"].astype(str)
    o["group"] = o["group"].astype(str)
    plate = o["plate"].to_numpy()
    is_ntc = (o["group"] == "NTC").to_numpy()

    # OXPHOS transcriptome score
    sym = a.var["symbol"].astype(str).to_numpy()
    idx = [int(np.where(sym == g)[0][0]) for g in OXPHOS if g in sym]
    ln = a.layers["lognorm"]
    ox = np.asarray(ln[:, idx].todense() if hasattr(ln, "todense") else ln[:, idx]).mean(1)

    cols = {
        "permito": "pheno_permito_dpsi_z",
        "mitomass": "pheno_mitomass_z",
        "intensity": "pheno_intensity_z",
        "area": "pheno_area_z",
        "pred_AUC": "vassay_pred_AUC",
        "pred_MB": "vassay_pred_MB",
        "ox": None,
    }
    z = {}
    for name, c in cols.items():
        raw = ox if name == "ox" else o[c].to_numpy(float)
        z[name] = batch_ntc_z(raw, plate, is_ntc)

    zdf = pd.DataFrame(z)
    zdf["group"] = o["group"].to_numpy()
    zdf["tox"] = o["tox_flag"].to_numpy()
    zdf["cat"] = o["category"].astype(str).to_numpy()
    clean = zdf[(~zdf["tox"]) & (zdf["cat"].isin(["Target", "PC"]))].dropna()

    # split-half reliability per modality
    print("=== per-modality split-half reliability (signal ceiling) ===")
    rel = {}
    for name in ["permito", "mitomass", "intensity", "area", "pred_AUC", "pred_MB", "ox"]:
        r, n = split_half(clean[["group", name]].rename(columns={name: "v"}), "v")
        rel[name] = r
        print(f"  {name:10s} r={r:+.3f} (n={n})")

    # target-level means
    tgt = clean.groupby("group").filter(lambda s: len(s) >= 4).groupby("group").mean(numeric_only=True)

    pairs = [
        ("permito", "pred_AUC"), ("area", "pred_AUC"), ("mitomass", "pred_MB"),
        ("intensity", "pred_MB"), ("ox", "pred_AUC"), ("ox", "mitomass"),
        ("ox", "permito"), ("mitomass", "pred_AUC"),
    ]
    print("\n=== cross-modal target-level correlation (n=%d) ===" % len(tgt))
    res = []
    for x, y in pairs:
        r = pearsonr(tgt[x], tgt[y])[0]
        rho = spearmanr(tgt[x], tgt[y])[0]
        res.append((f"{x} ↔ {y}", r, rho))
        print(f"  {x:9s} ↔ {y:9s}  Pearson={r:+.3f}  Spearman={rho:+.3f}")

    # figure: reliability bars + cross-modal heatmap
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = ["permito", "mitomass", "intensity", "area", "pred_AUC", "pred_MB", "ox"]
    axes[0].bar(names, [rel[n] for n in names], color="tab:green")
    axes[0].set_title("Per-modality split-half reliability")
    axes[0].set_ylabel("Pearson r (target signal)")
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].tick_params(axis="x", rotation=45)

    mods = ["ox", "permito", "mitomass", "intensity", "area", "pred_AUC", "pred_MB"]
    M = np.zeros((len(mods), len(mods)))
    for i, x in enumerate(mods):
        for j, y in enumerate(mods):
            M[i, j] = pearsonr(tgt[x], tgt[y])[0]
    im = axes[1].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(mods)))
    axes[1].set_xticklabels(mods, rotation=45, ha="right")
    axes[1].set_yticks(range(len(mods)))
    axes[1].set_yticklabels(mods)
    for i in range(len(mods)):
        for j in range(len(mods)):
            axes[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
    axes[1].set_title("Cross-modal target-level Pearson")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("Tri-modal consistency (transcriptome × C24 imaging × TMRM phenotype)")
    fig.tight_layout()
    p = figs / "multimodal_consistency.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\n[analyze] wrote {p}")

    tgt.to_csv(out / "multimodal_target_zscores.csv")
    print(f"[analyze] wrote {out / 'multimodal_target_zscores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
