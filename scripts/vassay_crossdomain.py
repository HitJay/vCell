#!/usr/bin/env python
"""P3 — cross-domain external validation of vAssay against drug-seq Seahorse.

The legacy vAssay models were trained on a compound/siRNA mix (264 wells, 9
plates). The real deployment domain is the drug-seq siRNA knockdown screen. This
script asks: do the **legacy vAssay predictions** actually track the *real
Seahorse* ground truth measured on a subset of drug-seq targets
(data/seahorse_vAssay_validation.csv)?

It compares, at target level (vs same-plate NTC %):
  * legacy vAssay pred_AUC  (already in seahorse_vAssay_validation.csv, historical)
  * current back-filled C24 pred_AUC (re-derived) — for reference

This is the honest "is the model useful on the application domain" check, and it
complements scripts/validate_seahorse.py (which focuses on the current pipeline).

Run: /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/vassay_crossdomain.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", default="data/seahorse_vAssay_validation.csv")
    ap.add_argument("--out", default="output/2026-06-11/vassay_review")
    args = ap.parse_args(argv)

    val = pd.read_csv(args.val)
    ev = val[(~val.target.isin(["NTC", "NoAdd"])) & val.vassay_AUC_pct.notna()].copy()
    g = ev.groupby("target").agg(
        seahorse=("seahorse_AUC_pct", "mean"),
        vassay=("vassay_AUC_pct", "mean"),
        needs_repeat=("needs_repeat", "max"),
    ).dropna()

    print("=== legacy vAssay vs ground-truth Seahorse (drug-seq domain) ===")
    print(g.round(3).to_string())
    out_rows = []
    for label, sub in [("all", g), ("excl_needs_repeat", g[g.needs_repeat == 0])]:
        if len(sub) < 4:
            continue
        r = pearsonr(sub.seahorse, sub.vassay)[0]
        rho = spearmanr(sub.seahorse, sub.vassay)[0]
        mae = float(np.abs(sub.seahorse - sub.vassay).mean())
        out_rows.append({"subset": label, "n": len(sub), "pearson": r,
                         "spearman": rho, "mae": mae})
        print(f"  {label:18s} n={len(sub):2d}  Pearson={r:+.3f}  "
              f"Spearman={rho:+.3f}  MAE={mae:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out / "crossdomain_metrics.csv", index=False)
    g.to_csv(out / "crossdomain_targets.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    clean = g[g.needs_repeat == 0]
    ax.scatter(clean.seahorse, clean.vassay, s=55, c="tab:blue",
               edgecolor="k", linewidth=0.4, label="clean")
    flagged = g[g.needs_repeat == 1]
    if len(flagged):
        ax.scatter(flagged.seahorse, flagged.vassay, s=55, c="tab:orange",
                   edgecolor="k", linewidth=0.4, label="needs_repeat")
    for t, row in g.iterrows():
        ax.annotate(t, (row.seahorse, row.vassay), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    lims = [0.2, 1.25]
    ax.plot(lims, lims, "--", color="grey", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Seahorse AUC% (ground truth)")
    ax.set_ylabel("legacy vAssay AUC% (predicted)")
    ax.set_title("Cross-domain check: legacy vAssay vs real Seahorse\n(drug-seq siRNA targets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "crossdomain_scatter.png", dpi=120)
    plt.close(fig)
    print(f"\n[crossdomain] wrote {out / 'crossdomain_scatter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
