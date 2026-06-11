#!/usr/bin/env python
"""Leakage-aware benchmark of the vAssay imaging → Seahorse models.

Covers the review action items:
  P1  leakage-aware CV (random vs group_plate vs group_treatment vs LOGO)
  P2  baseline matrix (channel × target × model × cv-scheme + mean baseline)
  P4  channel ranking under honest CV

Writes a tidy results CSV and a comparison figure to output/<date>/vassay_review/.

Run (TabPFN needs the env var):
    SCIPY_ARRAY_API=1 /data/user/QYJI/miniforge3/envs/cp3/bin/python \
        scripts/vassay_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.vassay import (  # noqa: E402
    baseline_metrics,
    load_vassay_csv,
    run_cv,
)

CHANNELS = ["C1", "C24", "C14", "C12"]
SCHEMES = ["random", "group_plate", "group_treatment"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/vassay_train")
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--targets", nargs="+", default=["AUC", "MB"])
    ap.add_argument("--models", nargs="+", default=["ridge", "tabpfn"])
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--aggregate", action="store_true",
                    help="collapse (plate,treatment) replicates into independent "
                         "units (removes label leakage).")
    ap.add_argument("--sirna-only", action="store_true",
                    help="keep only siRNA-knockdown treatments (deployment domain).")
    ap.add_argument("--out", default="output/2026-06-11/vassay_review")
    args = ap.parse_args(argv)

    ddir = Path(args.data_dir)
    rows = []
    for ch in args.channels:
        csv = ddir / f"train_{ch}.csv"
        if not csv.exists():
            print(f"[skip] {csv} missing")
            continue
        for tgt in args.targets:
            data = load_vassay_csv(csv, target=tgt, aggregate=args.aggregate,
                                   sirna_only=args.sirna_only)
            for scheme in args.schemes:
                base = baseline_metrics(data, scheme)
                rows.append({"channel": ch, "target": tgt, "model": "mean_baseline",
                             "scheme": scheme, "n": data.n,
                             **{f"{k}": v for k, v in base.items()}})
                for mdl in args.models:
                    res = run_cv(data, mdl, scheme)
                    rows.append({
                        "channel": ch, "target": tgt, "model": mdl, "scheme": scheme,
                        "n": data.n, **res.metrics_mean,
                        **{f"{k}_std": v for k, v in res.metrics_std.items()},
                    })
                    m = res.metrics_mean
                    print(f"{ch:4s} {tgt:3s} {mdl:7s} {scheme:16s} n={data.n:3d} "
                          f"R2={m['r2']:+.3f} r={m['pearson']:+.3f} "
                          f"rho={m['spearman']:+.3f}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_results.csv", index=False)
    print(f"\n[benchmark] wrote {out / 'benchmark_results.csv'}")

    _plot(df, out)
    return 0


def _plot(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["model"] == "tabpfn") & (df["target"] == "AUC")]
    if sub.empty:
        sub = df[(df["model"] == "ridge") & (df["target"] == "AUC")]
    if sub.empty:
        return
    schemes = ["random", "group_plate", "group_treatment"]
    chans = sorted(sub["channel"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / len(schemes)
    x = np.arange(len(chans))
    colors = {"random": "tab:red", "group_plate": "tab:orange",
              "group_treatment": "tab:green"}
    for i, sc in enumerate(schemes):
        vals = [sub[(sub.channel == c) & (sub.scheme == sc)]["pearson"].mean() for c in chans]
        ax.bar(x + i * width, vals, width, label=sc, color=colors.get(sc, None))
    ax.set_xticks(x + width)
    ax.set_xticklabels(chans)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Pearson r (out-of-fold)")
    ax.set_xlabel("imaging channel")
    ax.set_title("vAssay AUC prediction: random CV (leaky) vs grouped CV (honest)")
    ax.legend(title="CV scheme")
    fig.tight_layout()
    fig.savefig(out / "cv_scheme_comparison.png", dpi=120)
    plt.close(fig)
    print(f"[benchmark] wrote {out / 'cv_scheme_comparison.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
