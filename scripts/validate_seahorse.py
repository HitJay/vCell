#!/usr/bin/env python
"""Validate vAssay (C24 imaging -> TabPFN) predictions against ground-truth
Seahorse measurements for the subset of targets that were assayed on both.

Reads the hand-curated ground-truth table (data/seahorse_vAssay_validation.csv,
transcribed from image.png) and re-derives the *current* vAssay AUC% from the
freshly back-filled C24 readouts, then reports accuracy (Pearson / Spearman /
MAE) and writes a scatter plot comparing both the historical and current vAssay
predictions to the Seahorse truth.

Run with the scvi env:
    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/validate_seahorse.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

IMAGE_BASE = "/NNRCC_Image/processed_data/UHYG/2025"


def current_vassay_auc_pct(adata_path: str) -> pd.Series:
    """Per-target current vAssay AUC%, batch-normalised to same-plate NTC."""
    import anndata as ad

    o = ad.read_h5ad(adata_path).obs
    o = o[["group", "well", "plate", "tmrm_operetta_data_file_name"]].copy()
    o["imgplate"] = o["tmrm_operetta_data_file_name"].astype(str)

    out = []
    for p in o["imgplate"].unique():
        f = glob.glob(f"{IMAGE_BASE}/{p}/csv/*vAssay_readout_C24.csv")
        if not f:
            continue
        d = pd.read_csv(f[0])[["ImgID", "pred_AUC"]].rename(columns={"ImgID": "well"})
        d["imgplate"] = p
        out.append(d)
    r = pd.concat(out).merge(o, on=["imgplate", "well"])

    r["auc_pct"] = np.nan
    for pl in r["plate"].unique():
        m = r["plate"] == pl
        ntc = m & (r["group"] == "NTC")
        ref = r.loc[ntc, "pred_AUC"].mean() if ntc.any() else r.loc[m, "pred_AUC"].mean()
        r.loc[m, "auc_pct"] = r.loc[m, "pred_AUC"] / ref
    return r.groupby("group", observed=True)["auc_pct"].mean()


def _metrics(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": len(x),
        "pearson": pearsonr(x, y)[0],
        "pearson_p": pearsonr(x, y)[1],
        "spearman": spearmanr(x, y)[0],
        "mae": float(np.abs(x - y).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", default="data/seahorse_vAssay_validation.csv")
    ap.add_argument("--adata", default="data/drug-seq/adata.h5ad")
    ap.add_argument("--out", default="output/2026-06-10")
    args = ap.parse_args(argv)

    val = pd.read_csv(args.val)
    targets = sorted(set(val["target"]) - {"NTC", "NoAdd"})
    sh = val[val.target.isin(targets)].groupby("target")["seahorse_AUC_pct"].mean()
    hist = val[val.target.isin(targets)].groupby("target")["vassay_AUC_pct"].mean()
    cur = current_vassay_auc_pct(args.adata)

    cmp = pd.DataFrame({"seahorse": sh, "vassay_hist": hist,
                        "vassay_current": cur}).dropna(subset=["seahorse"])

    print("=== target-level: Seahorse truth vs vAssay ===")
    print(cmp.round(3).to_string())
    print()
    results = {}
    for col in ["vassay_hist", "vassay_current"]:
        d = cmp.dropna(subset=[col])
        if len(d) >= 4:
            m = _metrics(d["seahorse"].to_numpy(), d[col].to_numpy())
            results[col] = m
            print(f"  {col:16s} n={m['n']:2d}  Pearson={m['pearson']:.3f} "
                  f"(p={m['pearson_p']:.3f})  Spearman={m['spearman']:.3f}  "
                  f"MAE={m['mae']:.3f}")

    # scatter plot
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "vassay_hist", "historical vAssay"),
        (axes[1], "vassay_current", "current (back-filled) vAssay"),
    ]:
        d = cmp.dropna(subset=[col])
        ax.scatter(d["seahorse"], d[col], s=45, c="tab:blue", edgecolor="k", linewidth=0.4)
        for t, row in d.iterrows():
            ax.annotate(t, (row["seahorse"], row[col]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        lims = [0.2, 1.25]
        ax.plot(lims, lims, "--", color="grey", lw=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Seahorse AUC% (truth, vs NTC)")
        ax.set_ylabel("vAssay AUC% (predicted)")
        m = results.get(col, {})
        ax.set_title(f"{title}\nn={m.get('n','-')}  r={m.get('pearson',float('nan')):.2f}  "
                     f"ρ={m.get('spearman',float('nan')):.2f}  MAE={m.get('mae',float('nan')):.2f}")
        ax.legend(loc="upper left")
    fig.suptitle("vAssay (C24 imaging → Seahorse) validation vs ground-truth Seahorse")
    fig.tight_layout()
    p = figs / "seahorse_validation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\n[validate] wrote {p}")

    cmp.to_csv(out / "seahorse_validation_targets.csv")
    print(f"[validate] wrote {out / 'seahorse_validation_targets.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
