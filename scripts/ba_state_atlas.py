#!/usr/bin/env python
"""Phase 1 for the B+A route: EE mitochondrial state atlas and hit calling.

Inputs are the main-line-D processed target/well summaries. The script builds a
transparent target-level atlas before any complex modelling:

* MitoTracker-resolved state classes from per-mito dPsi and mito mass axes.
* Bootstrap confidence intervals and split-half direction checks from wells.
* A conservative consensus score that rewards phenotype strength, KD quality,
  reproducibility and cross-modal support, while penalising toxicity/QC issues.
* Candidate priority table and figures for review.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PHENO_COLS = {
    "permito": "pheno_permito_dpsi_z",
    "mitomass": "pheno_mitomass_z",
    "area": "pheno_area_z",
    "intensity": "pheno_intensity_z",
}

STATE_COLORS = {
    "uncoupler_like": "#d95f02",
    "mixed_uncoupling_biogenesis": "#7570b3",
    "biogenesis_like": "#1b9e77",
    "energizer_like": "#66a61e",
    "toxic_collapse": "#b2182b",
    "neutral_or_uncertain": "#8c8c8c",
}


@dataclass(frozen=True)
class ScoreConfig:
    phenotype_clip: float = 10.0
    state_threshold: float = 3.0
    tox_soft: float = 0.2
    tox_hard: float = 0.5
    qc_hard: float = 0.4


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    if len(finite) == 1 or n_boot <= 0:
        return float(finite[0]), float(finite[0])
    idx = rng.integers(0, len(finite), size=(n_boot, len(finite)))
    means = finite[idx].mean(axis=1)
    lo, hi = np.percentile(means, [5, 95])
    return float(lo), float(hi)


def split_half(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, bool]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        return float("nan"), float("nan"), False
    idx = rng.permutation(len(finite))
    half = len(finite) // 2
    left = float(finite[idx[:half]].mean())
    right = float(finite[idx[half:]].mean())
    consistent = (left == 0 and right == 0) or (np.sign(left) == np.sign(right))
    return left, right, bool(consistent)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def classify_state(row: pd.Series, cfg: ScoreConfig) -> str:
    permito = safe_float(row["permito"])
    mitomass = safe_float(row["mitomass"])
    area = safe_float(row["area"])
    tox_rate = safe_float(row.get("tox_rate", 0.0))
    qc_rate = safe_float(row.get("qc_fail_rate", 0.0))

    if tox_rate >= cfg.tox_hard or (tox_rate >= cfg.tox_soft and permito <= -cfg.state_threshold and area <= -cfg.state_threshold):
        return "toxic_collapse"
    if qc_rate >= cfg.qc_hard:
        return "neutral_or_uncertain"
    if permito >= 2.0 and mitomass >= 2.0:
        return "energizer_like"
    if permito <= -cfg.state_threshold and mitomass >= cfg.state_threshold:
        return "mixed_uncoupling_biogenesis"
    if permito <= -cfg.state_threshold or area <= -cfg.state_threshold:
        return "uncoupler_like"
    if mitomass >= cfg.state_threshold:
        return "biogenesis_like"
    return "neutral_or_uncertain"


def dominant_axis(row: pd.Series) -> str:
    vals = {
        axis: abs(safe_float(row[axis], default=float("nan")))
        for axis in ["permito", "mitomass", "area"]
        if np.isfinite(safe_float(row[axis], default=float("nan")))
    }
    if not vals:
        return "permito"
    return max(vals, key=vals.get)


def confidence_excludes_zero(row: pd.Series, axis: str) -> bool:
    lo = row.get(f"{axis}_ci90_low", np.nan)
    hi = row.get(f"{axis}_ci90_high", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return bool(lo > 0 or hi < 0)


def score_row(row: pd.Series, cfg: ScoreConfig) -> float:
    phenotype_strength = min(
        cfg.phenotype_clip,
        max(abs(row["permito"]), abs(row["mitomass"]), abs(row["area"])),
    )
    kd_bonus = {"strong": 2.0, "weak": 1.0, "failed": -3.0, "unknown": -1.0}.get(
        str(row.get("kd_tier", "")).lower(), 0.0
    )
    consistency_bonus = 1.5 if row.get("dominant_ci_excludes_zero", False) else 0.0
    if row.get("dominant_split_consistent", False):
        consistency_bonus += 0.75

    crossmodal_bonus = 0.0
    for col in ["pred_AUC", "pred_MB", "ox"]:
        if col in row and np.isfinite(row[col]) and abs(row[col]) >= 2.0:
            crossmodal_bonus += 0.35
    crossmodal_bonus = min(crossmodal_bonus, 1.0)

    toxicity_penalty = min(6.0, safe_float(row.get("tox_rate", 0.0)) * 8.0)
    qc_penalty = min(3.0, safe_float(row.get("qc_fail_rate", 0.0)) * 5.0)
    toxic_state_penalty = 3.0 if row.get("state_class") == "toxic_collapse" else 0.0
    return float(phenotype_strength + kd_bonus + consistency_bonus + crossmodal_bonus - toxicity_penalty - qc_penalty - toxic_state_penalty)


def recommend(row: pd.Series) -> str:
    if row.get("category") == "PC":
        return "positive_control_reference"
    if row.get("category") != "Target":
        return "non_target_reference"
    if row["state_class"] == "toxic_collapse" or safe_float(row.get("tox_rate", 0.0)) >= 0.5:
        return "deprioritize_toxic"
    if row["state_class"] == "neutral_or_uncertain" and row["phenotype_strength"] < 3.0:
        return "monitor_neutral"
    if str(row.get("kd_tier", "")).lower() == "failed" and row["phenotype_strength"] >= 5.0 and safe_float(row.get("tox_rate", 0.0)) < 0.25:
        return "kd_rescue_or_repeat"
    if (
        row["consensus_score"] >= 8.0
        and safe_float(row.get("tox_rate", 0.0)) <= 0.2
        and safe_float(row.get("qc_fail_rate", 0.0)) <= 0.25
        and str(row.get("kd_tier", "")).lower() in {"strong", "weak"}
        and row.get("dominant_ci_excludes_zero", False)
        and row.get("n_clean_wells", 0) >= 4
    ):
        return "tier1_immediate_validation"
    if (
        row["consensus_score"] >= 5.0
        and safe_float(row.get("tox_rate", 0.0)) <= 0.35
        and str(row.get("kd_tier", "")).lower() != "failed"
        and row.get("n_clean_wells", 0) >= 3
    ):
        return "tier2_secondary_review"
    if str(row.get("kd_tier", "")).lower() == "failed":
        return "deprioritize_failed_kd"
    return "tier3_low_priority"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def build_atlas(
    targets: pd.DataFrame,
    wells: pd.DataFrame,
    multimodal: pd.DataFrame | None,
    seed: int,
    n_boot: int,
    cfg: ScoreConfig,
) -> pd.DataFrame:
    wells = wells.copy()
    wells["qc_fail"] = as_bool(wells["qc_fail"])
    wells["tox_flag"] = as_bool(wells["tox_flag"])
    clean = wells[(~wells["qc_fail"]) & (~wells["tox_flag"]) & (wells["category"].isin(["Target", "PC", "NC"]))]

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for group, sub in clean.groupby("group"):
        rec: dict[str, object] = {"group": group, "n_clean_wells": int(len(sub))}
        for short, col in PHENO_COLS.items():
            values = sub[col].to_numpy(dtype=float)
            rec[short] = float(np.nanmean(values)) if len(values) else float("nan")
            lo, hi = bootstrap_ci(values, rng, n_boot)
            rec[f"{short}_ci90_low"] = lo
            rec[f"{short}_ci90_high"] = hi
            left, right, consistent = split_half(values, rng)
            rec[f"{short}_split1"] = left
            rec[f"{short}_split2"] = right
            rec[f"{short}_split_consistent"] = consistent
        rows.append(rec)
    atlas = pd.DataFrame(rows)

    keep_cols = [
        "group",
        "category",
        "n_wells",
        "n_batches",
        "tox_rate",
        "qc_fail_rate",
        "kd_frac_drop",
        "kd_tier",
        "n_kd",
        "ee_hit_candidate",
        "ee_hit_direction",
        *PHENO_COLS.values(),
    ]
    keep_cols = [col for col in keep_cols if col in targets.columns]
    atlas = targets[keep_cols].merge(atlas, on="group", how="left")
    for short, original in PHENO_COLS.items():
        if short not in atlas or atlas[short].isna().all():
            atlas[short] = atlas[original]
        else:
            atlas[short] = atlas[short].fillna(atlas[original])

    if multimodal is not None:
        mm_cols = [col for col in ["group", "pred_AUC", "pred_MB", "ox"] if col in multimodal.columns]
        atlas = atlas.merge(multimodal[mm_cols], on="group", how="left")

    atlas["dominant_axis"] = atlas.apply(dominant_axis, axis=1)
    atlas["dominant_ci_excludes_zero"] = atlas.apply(lambda row: confidence_excludes_zero(row, row["dominant_axis"]), axis=1)
    atlas["dominant_split_consistent"] = atlas.apply(
        lambda row: bool(row.get(f"{row['dominant_axis']}_split_consistent", False)), axis=1
    )
    atlas["phenotype_strength"] = atlas[["permito", "mitomass", "area"]].abs().max(axis=1).clip(upper=cfg.phenotype_clip)
    atlas["state_class"] = atlas.apply(lambda row: classify_state(row, cfg), axis=1)
    atlas["consensus_score"] = atlas.apply(lambda row: score_row(row, cfg), axis=1)
    atlas["recommendation"] = atlas.apply(recommend, axis=1)
    return atlas.sort_values(["consensus_score", "phenotype_strength"], ascending=False).reset_index(drop=True)


def write_figures(atlas: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = atlas[atlas["category"].isin(["Target", "PC", "NC"])].copy()
    for state, sub in plot_df.groupby("state_class"):
        ax.scatter(
            sub["mitomass"],
            sub["permito"],
            s=28 + 12 * sub["phenotype_strength"].fillna(0),
            c=STATE_COLORS.get(state, "#8c8c8c"),
            alpha=0.75,
            label=state,
            edgecolor="black" if state == "toxic_collapse" else "none",
            linewidth=0.4,
        )
    for group in ["BAM15", "MK8722", "ATP5B", "SLC25A4", "PSMC3"]:
        sub = plot_df[plot_df["group"] == group]
        if not sub.empty:
            row = sub.iloc[0]
            ax.annotate(group, (row["mitomass"], row["permito"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    top_targets = atlas[
        (atlas["category"] == "Target")
        & (atlas["recommendation"].isin(["tier1_immediate_validation", "tier2_secondary_review"]))
    ].head(top_n)
    for _, row in top_targets.head(12).iterrows():
        ax.annotate(row["group"], (row["mitomass"], row["permito"]), xytext=(4, -8), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="grey", lw=0.7)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("Mito mass / biogenesis z (ch4/ch1)")
    ax.set_ylabel("Per-mito dPsi z (ch2/ch4)")
    ax.set_title("B+A Phase 1: mitochondrial state atlas")
    ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "mitochondrial_state_atlas.png", dpi=150)
    plt.close(fig)

    cand = atlas[atlas["category"] == "Target"].head(30).iloc[::-1]
    colors = [STATE_COLORS.get(state, "#8c8c8c") for state in cand["state_class"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(cand["group"], cand["consensus_score"], color=colors)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("Consensus score")
    ax.set_title("Top target candidates by transparent consensus score")
    fig.tight_layout()
    fig.savefig(figs / "candidate_score_waterfall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    target = atlas[atlas["category"] == "Target"]
    ax.scatter(
        target["tox_rate"],
        target["phenotype_strength"],
        c=[STATE_COLORS.get(state, "#8c8c8c") for state in target["state_class"]],
        alpha=0.75,
    )
    ax.axvline(0.2, color="orange", ls="--", lw=0.8, label="tox_rate 0.2")
    ax.axvline(0.5, color="red", ls="--", lw=0.8, label="tox_rate 0.5")
    ax.set_xlabel("Toxic well rate")
    ax.set_ylabel("Phenotype strength")
    ax.set_title("Toxicity penalty check")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "toxicity_vs_phenotype_strength.png", dpi=150)
    plt.close(fig)


def write_summary(atlas: pd.DataFrame, out_dir: Path) -> None:
    target = atlas[atlas["category"] == "Target"].copy()
    state_counts = target["state_class"].value_counts().rename_axis("state_class").reset_index(name="n_targets")
    rec_counts = target["recommendation"].value_counts().rename_axis("recommendation").reset_index(name="n_targets")
    top_cols = [
        "group",
        "state_class",
        "recommendation",
        "consensus_score",
        "permito",
        "mitomass",
        "area",
        "kd_tier",
        "tox_rate",
        "dominant_ci_excludes_zero",
    ]
    top = target[
        target["recommendation"].isin(["tier1_immediate_validation", "tier2_secondary_review", "kd_rescue_or_repeat"])
    ][top_cols].head(20)
    tier1 = target[target["recommendation"] == "tier1_immediate_validation"][top_cols]

    md = f"""# B+A Phase 1 - Mitochondrial State Atlas

Generated from `data/processed/targets_summary.csv` and `data/processed/wells_annotation.csv`.

## Summary

- Target rows scored: **{len(target)}**
- Tier 1 immediate-validation candidates: **{len(tier1)}**
- Scoring is phenotype-first, with KD/reproducibility bonuses and toxicity/QC penalties.
- C24/vAssay and OXPHOS z-scores are only auxiliary support; they are not allowed to dominate the hit score.

## Target State Counts

{markdown_table(state_counts)}

## Recommendation Counts

{markdown_table(rec_counts)}

## Top Candidates For Review

{markdown_table(top.round(2))}

## Figures

- `figs/mitochondrial_state_atlas.png` - per-mito dPsi vs mito mass state map.
- `figs/candidate_score_waterfall.png` - top target consensus-score waterfall.
- `figs/toxicity_vs_phenotype_strength.png` - toxicity penalty sanity check.

## Reading Rules

- `tier1_immediate_validation`: phenotype strong, KD credible, low toxicity/QC burden, and dominant-axis CI excludes zero.
- `tier2_secondary_review`: promising but needs manual review before validation.
- `kd_rescue_or_repeat`: phenotype is strong but KD failed or is suspect; do not treat as a clean negative/positive yet.
- `deprioritize_toxic`: phenotype is likely confounded by cell loss.
"""
    (out_dir / "B_A_hit_calling_summary.md").write_text(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="data/processed/targets_summary.csv")
    parser.add_argument("--wells", default="data/processed/wells_annotation.csv")
    parser.add_argument("--multimodal", default="output/2026-06-10/multimodal_target_zscores.csv")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)

    targets = pd.read_csv(args.targets)
    wells = pd.read_csv(args.wells, index_col=0)
    multimodal = pd.read_csv(args.multimodal) if args.multimodal and Path(args.multimodal).exists() else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ScoreConfig()
    atlas = build_atlas(targets, wells, multimodal, args.seed, args.n_boot, cfg)
    atlas.to_csv(out_dir / "B_A_state_atlas.csv", index=False)
    priority = atlas[atlas["category"] == "Target"].sort_values("consensus_score", ascending=False)
    priority.to_csv(out_dir / "B_A_candidate_priority.csv", index=False)
    write_figures(atlas, out_dir, args.top_n)
    write_summary(atlas, out_dir)

    tier1 = int((priority["recommendation"] == "tier1_immediate_validation").sum())
    tier2 = int((priority["recommendation"] == "tier2_secondary_review").sum())
    print(f"[ba_state_atlas] wrote {out_dir / 'B_A_state_atlas.csv'}")
    print(f"[ba_state_atlas] wrote {out_dir / 'B_A_candidate_priority.csv'}")
    print(f"[ba_state_atlas] tier1={tier1} tier2={tier2} total_targets={len(priority)}")
    print(f"[ba_state_atlas] figures under {out_dir / 'figs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())