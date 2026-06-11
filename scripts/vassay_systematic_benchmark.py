#!/usr/bin/env python
"""Systematic vAssay benchmark — the honest performance picture.

Produces a single tidy table + figure covering three nested questions:

  1. RAW (264 image rows, no aggregation)      — reproduces the leaky legacy R².
  2. AGGREGATED (one row per plate×treatment)   — removes field-replicate label
     leakage; the remaining gap to grouped CV is the plate/treatment effect.
  3. siRNA-DOMAIN (aggregated + compounds dropped) — the deployment domain;
     evaluated with leave-one-target-out (LOTO, pooled OOF).

For each it reports random vs grouped CV and the mean baseline, so the reader
sees exactly how much of the headline number is leakage vs real signal.

Run:
    SCIPY_ARRAY_API=1 /data/user/QYJI/miniforge3/envs/cp3/bin/python \
        scripts/vassay_systematic_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.vassay import baseline_metrics, load_vassay_csv, run_cv  # noqa: E402

CHANNELS = ["C1", "C24", "C14", "C12"]


def _row(setting, ch, model, scheme, res_or_base, n, is_base=False):
    m = res_or_base if is_base else res_or_base.metrics_mean
    return {"setting": setting, "channel": ch, "model": model, "scheme": scheme,
            "n": n, "r2": m["r2"], "pearson": m["pearson"],
            "spearman": m["spearman"], "mae": m["mae"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/vassay_train")
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--target", default="AUC")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--out", default="output/2026-06-11/vassay_systematic")
    args = ap.parse_args(argv)

    ddir = Path(args.data_dir)
    rows = []
    settings = [
        ("raw", dict(aggregate=False, sirna_only=False), ["random", "group_plate", "group_treatment"]),
        ("aggregated", dict(aggregate=True, sirna_only=False), ["random", "group_plate", "group_treatment"]),
        ("sirna_domain", dict(aggregate=True, sirna_only=True), ["random", "logo_treatment"]),
    ]
    for setting, kw, schemes in settings:
        for ch in args.channels:
            csv = ddir / f"train_{ch}.csv"
            if not csv.exists():
                continue
            d = load_vassay_csv(csv, target=args.target, **kw)
            for scheme in schemes:
                rows.append(_row(setting, ch, "mean_baseline", scheme,
                                 baseline_metrics(d, scheme), d.n, is_base=True))
                res = run_cv(d, args.model, scheme)
                rows.append(_row(setting, ch, args.model, scheme, res, d.n))
                m = res.metrics_mean
                print(f"{setting:13s} {ch:4s} {scheme:16s} n={d.n:3d} "
                      f"R2={m['r2']:+.3f} r={m['pearson']:+.3f} rho={m['spearman']:+.3f}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "systematic_benchmark.csv", index=False)
    print(f"\n[systematic] wrote {out / 'systematic_benchmark.csv'}")
    _plot(df, args.model, args.target, out)
    return 0


def _plot(df, model, target, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = df[df.model == model]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    panels = [
        ("raw", ["random", "group_plate", "group_treatment"], "RAW (leaky)"),
        ("aggregated", ["random", "group_plate", "group_treatment"], "AGGREGATED (deleaked replicates)"),
        ("sirna_domain", ["random", "logo_treatment"], "siRNA DOMAIN (LOTO)"),
    ]
    colors = {"random": "tab:red", "group_plate": "tab:orange",
              "group_treatment": "tab:green", "logo_treatment": "tab:green"}
    chans = sorted(md.channel.unique())
    for ax, (setting, schemes, title) in zip(axes, panels):
        sub = md[md.setting == setting]
        x = np.arange(len(chans))
        w = 0.8 / len(schemes)
        for i, sc in enumerate(schemes):
            vals = [sub[(sub.channel == c) & (sub.scheme == sc)]["spearman"].mean() for c in chans]
            ax.bar(x + i * w, vals, w, label=sc, color=colors.get(sc))
        ax.set_xticks(x + w * (len(schemes) - 1) / 2)
        ax.set_xticklabels(chans)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("channel")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"Spearman ({target}, out-of-fold)")
    fig.suptitle(f"vAssay {target} — leakage decomposition ({model}): "
                 "random CV is inflated; grouped/LOTO is honest", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "leakage_decomposition.png", dpi=120)
    plt.close(fig)
    print(f"[systematic] wrote {out / 'leakage_decomposition.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
