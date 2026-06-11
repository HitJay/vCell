#!/usr/bin/env python
"""P4/P5 — summarise the leakage-aware benchmark into an honest channel ranking
and a retrain decision.

Reads benchmark_results.csv (from scripts/vassay_benchmark.py) and prints, for
the deployment-relevant CV scheme (group_treatment = generalize to unseen
perturbations), the channel ranking by Spearman — the metric that matters for
EE hit ranking. Flags whether any config beats the mean baseline meaningfully.

Run: /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/vassay_summary.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="output/2026-06-11/vassay_review/benchmark_results.csv")
    ap.add_argument("--deploy-scheme", default="group_treatment")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.results)
    print("=" * 64)
    print("vAssay honest benchmark summary")
    print("=" * 64)

    # 1) leakage magnitude: random vs deploy scheme (per channel, AUC, best model)
    print("\n[1] Leakage magnitude (AUC, best non-baseline model)")
    md = df[df.model != "mean_baseline"]
    for ch in sorted(md.channel.unique()):
        rnd = md[(md.channel == ch) & (md.target == "AUC") & (md.scheme == "random")]
        dep = md[(md.channel == ch) & (md.target == "AUC") & (md.scheme == args.deploy_scheme)]
        if rnd.empty or dep.empty:
            continue
        rr = rnd.pearson.max()
        dd = dep.pearson.max()
        print(f"  {ch:4s}  random r={rr:+.3f}  ->  {args.deploy_scheme} r={dd:+.3f}  "
              f"(drop {rr - dd:+.3f})")

    # 2) honest channel ranking by Spearman under deploy scheme
    print(f"\n[2] Channel ranking — AUC, {args.deploy_scheme}, by Spearman")
    dep = md[(md.target == "AUC") & (md.scheme == args.deploy_scheme)]
    rank = dep.groupby("channel")[["pearson", "spearman", "r2"]].max().sort_values(
        "spearman", ascending=False)
    print(rank.round(3).to_string())

    # 3) vs baseline
    print(f"\n[3] vs mean baseline ({args.deploy_scheme})")
    base = df[(df.model == "mean_baseline") & (df.target == "AUC") &
              (df.scheme == args.deploy_scheme)]
    for ch in sorted(dep.channel.unique()):
        b = base[base.channel == ch]
        m = dep[dep.channel == ch]
        if b.empty or m.empty:
            continue
        print(f"  {ch:4s}  model spearman={m.spearman.max():+.3f}  "
              f"baseline spearman={b.spearman.max():+.3f}")

    # 4) retrain verdict
    print("\n[4] Verdict")
    best = rank.index[0] if len(rank) else None
    best_rho = rank.spearman.max() if len(rank) else float("nan")
    print(f"  Best generalizing channel (AUC): {best} (Spearman {best_rho:.3f})")
    if best_rho >= 0.5:
        print("  -> Usable for *ranking* EE hits (trust Spearman, not absolute value).")
        print("  -> Retrain recommended: fix this channel, report grouped-CV metric.")
    else:
        print("  -> Weak generalization; treat predictions as noisy. Need more")
        print("     independent plates / siRNA-domain data before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
