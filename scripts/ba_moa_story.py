#!/usr/bin/env python
"""Build the B+A cross-modal MoA story package.

This script turns the existing cross-modal MoA outputs into a story-oriented
deliverable: prespecified state-level tests, representative target evidence
strips, compact figures and a narrative markdown report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402


STATE_ORDER = [
    "neutral_or_uncertain",
    "uncoupler_like",
    "mixed_uncoupling_biogenesis",
    "biogenesis_like",
    "toxic_collapse",
]

STATE_COLORS = {
    "neutral_or_uncertain": "#8c8c8c",
    "uncoupler_like": "#d95f02",
    "mixed_uncoupling_biogenesis": "#7570b3",
    "biogenesis_like": "#1b9e77",
    "toxic_collapse": "#b2182b",
}

STORY_FEATURES = [
    "permito",
    "mitomass",
    "area",
    "tox_rate",
    "conn_PSMC3",
    "path_OXPHOS_ETC",
    "path_MITO_BIOGENESIS",
    "path_ISR_ER_STRESS",
    "path_PROTEOSTASIS_AUTOPHAGY",
    "path_APOPTOSIS_TOXICITY",
]

FEATURE_LABELS = {
    "permito": "per-mito dPsi",
    "mitomass": "mito mass",
    "area": "TMRM area",
    "tox_rate": "tox rate",
    "conn_PSMC3": "PSMC3 conn.",
    "path_OXPHOS_ETC": "OXPHOS",
    "path_MITO_BIOGENESIS": "mito biogenesis genes",
    "path_ISR_ER_STRESS": "ISR/ER stress",
    "path_PROTEOSTASIS_AUTOPHAGY": "proteostasis/autophagy",
    "path_APOPTOSIS_TOXICITY": "apoptosis/toxicity",
}

REPRESENTATIVE_TARGETS = [
    "BAM15",
    "MK8722",
    "PSMC3",
    "DDI2",
    "G6PC",
    "TM6SF2",
    "DGAT2",
    "SLC39A11",
    "TAGLN",
    "NOTCH2",
    "SLC12A8",
    "INO80E",
]


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


def test_feature(df: pd.DataFrame, contrast: str, case_states: list[str], ref_states: list[str], feature: str) -> dict[str, object]:
    case = df[df["state_class"].isin(case_states)][feature].dropna().to_numpy(dtype=float)
    ref = df[df["state_class"].isin(ref_states)][feature].dropna().to_numpy(dtype=float)
    if len(case) < 3 or len(ref) < 3:
        pvalue = np.nan
        statistic = np.nan
    else:
        statistic, pvalue = mannwhitneyu(case, ref, alternative="two-sided")
    return {
        "contrast": contrast,
        "case_states": ";".join(case_states),
        "reference_states": ";".join(ref_states),
        "feature": feature,
        "feature_label": FEATURE_LABELS.get(feature, feature),
        "n_case": int(len(case)),
        "n_reference": int(len(ref)),
        "case_mean": float(np.nanmean(case)) if len(case) else np.nan,
        "reference_mean": float(np.nanmean(ref)) if len(ref) else np.nan,
        "effect_mean_diff": float(np.nanmean(case) - np.nanmean(ref)) if len(case) and len(ref) else np.nan,
        "case_median": float(np.nanmedian(case)) if len(case) else np.nan,
        "reference_median": float(np.nanmedian(ref)) if len(ref) else np.nan,
        "effect_median_diff": float(np.nanmedian(case) - np.nanmedian(ref)) if len(case) and len(ref) else np.nan,
        "mannwhitney_u": statistic,
        "p_value": pvalue,
    }


def run_story_tests(table: pd.DataFrame) -> pd.DataFrame:
    target = table[table["category"].eq("Target")].copy()
    tests = []
    specs = [
        (
            "toxic_collapse_vs_all_other_states",
            ["toxic_collapse"],
            ["neutral_or_uncertain", "uncoupler_like", "mixed_uncoupling_biogenesis", "biogenesis_like"],
            ["tox_rate", "conn_PSMC3", "path_ISR_ER_STRESS", "path_PROTEOSTASIS_AUTOPHAGY", "path_APOPTOSIS_TOXICITY", "permito", "area"],
        ),
        (
            "uncoupling_states_vs_neutral",
            ["uncoupler_like", "mixed_uncoupling_biogenesis"],
            ["neutral_or_uncertain"],
            ["permito", "area", "mitomass", "path_OXPHOS_ETC", "path_MITO_BIOGENESIS", "path_PROTEOSTASIS_AUTOPHAGY"],
        ),
        (
            "biogenesis_like_vs_neutral",
            ["biogenesis_like"],
            ["neutral_or_uncertain"],
            ["mitomass", "permito", "area", "path_MITO_BIOGENESIS", "path_FAO_LIPID", "conn_SLC25A4"],
        ),
        (
            "mixed_vs_uncoupler_only",
            ["mixed_uncoupling_biogenesis"],
            ["uncoupler_like"],
            ["mitomass", "permito", "area", "path_OXPHOS_ETC", "path_MITO_BIOGENESIS"],
        ),
    ]
    for contrast, case_states, ref_states, features in specs:
        for feature in features:
            if feature in target:
                tests.append(test_feature(target, contrast, case_states, ref_states, feature))
    out = pd.DataFrame(tests)
    out["fdr"] = fdr_bh(out["p_value"].to_numpy())
    out["neg_log10_fdr"] = -np.log10(out["fdr"].clip(lower=1e-300))
    return out.sort_values(["contrast", "fdr", "feature"])


def plot_state_boxplots(table: pd.DataFrame, out: Path) -> None:
    target = table[table["category"].eq("Target")].copy()
    features = [
        "permito",
        "mitomass",
        "area",
        "conn_PSMC3",
        "path_OXPHOS_ETC",
        "path_MITO_BIOGENESIS",
        "path_PROTEOSTASIS_AUTOPHAGY",
        "path_APOPTOSIS_TOXICITY",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    states = [s for s in STATE_ORDER if s in set(target["state_class"])]
    for ax, feature in zip(axes, features):
        values = [target[target["state_class"].eq(state)][feature].dropna().to_numpy(dtype=float) for state in states]
        bp = ax.boxplot(values, patch_artist=True, showfliers=False)
        for patch, state in zip(bp["boxes"], states):
            patch.set_facecolor(STATE_COLORS.get(state, "#cccccc"))
            patch.set_alpha(0.7)
        for state_idx, state in enumerate(states, start=1):
            y = target[target["state_class"].eq(state)][feature].dropna().to_numpy(dtype=float)
            x = np.full(len(y), state_idx) + np.random.default_rng(0).normal(0, 0.035, size=len(y))
            ax.scatter(x, y, s=12, color="black", alpha=0.35, linewidth=0)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(FEATURE_LABELS.get(feature, feature), fontsize=10)
        ax.set_xticks(np.arange(1, len(states) + 1))
        ax.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=7, rotation=0)
    fig.suptitle("State-level MoA evidence: phenotype, toxicity, reference connectivity and pathways")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def select_representatives(table: pd.DataFrame) -> pd.DataFrame:
    reps = table[table["group"].isin(REPRESENTATIVE_TARGETS)].copy()
    if reps.empty:
        return reps
    order = [g for g in REPRESENTATIVE_TARGETS if g in set(reps["group"])]
    reps["_order"] = reps["group"].map({g: i for i, g in enumerate(order)})
    return reps.sort_values("_order").drop(columns="_order")


def plot_evidence_strips(reps: pd.DataFrame, out: Path) -> pd.DataFrame:
    features = [
        "permito",
        "mitomass",
        "area",
        "tox_rate",
        "conn_BAM15",
        "conn_MK8722",
        "conn_PSMC3",
        "path_OXPHOS_ETC",
        "path_MITO_BIOGENESIS",
        "path_PROTEOSTASIS_AUTOPHAGY",
        "path_APOPTOSIS_TOXICITY",
    ]
    data = reps.set_index("group")[features].astype(float)
    scaled = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, np.nan)
    scaled = scaled.clip(-2.5, 2.5)
    fig, ax = plt.subplots(figsize=(15, max(4.5, 0.42 * len(scaled))))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([FEATURE_LABELS.get(f, f).replace("/", "/\n").replace(" ", "\n") for f in features], rotation=0, fontsize=8)
    labels = [f"{idx} ({reps.set_index('group').loc[idx, 'state_class']})" for idx in scaled.index]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            value = data.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("Representative target evidence strips (raw value text, column-z color)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(pad=1.1)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return data.reset_index()


def top_volcano_genes(volcano: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if volcano.empty:
        return volcano
    rows = []
    for contrast, sub in volcano.groupby("contrast"):
        ranked = sub.assign(rank_score=sub["effect_ntc_z"].abs() * sub["neg_log10_fdr"].fillna(0))
        rows.append(ranked.sort_values("rank_score", ascending=False).head(n))
    return pd.concat(rows, ignore_index=True).drop(columns=["rank_score"])


def format_test_table(tests: pd.DataFrame, contrast: str, n: int = 8) -> str:
    sub = tests[tests["contrast"].eq(contrast)].copy()
    if sub.empty:
        return "_No tests available._"
    cols = ["feature_label", "case_mean", "reference_mean", "effect_mean_diff", "p_value", "fdr"]
    return sub[cols].head(n).round(4).to_markdown(index=False)


def write_story(
    table: pd.DataFrame,
    tests: pd.DataFrame,
    reps: pd.DataFrame,
    top_genes: pd.DataFrame,
    out_dir: Path,
) -> None:
    target = table[table["category"].eq("Target")]
    state_counts = target["state_class"].value_counts().reindex(STATE_ORDER).dropna().astype(int)
    rep_cols = ["group", "state_class", "kd_tier", "tox_rate", "permito", "mitomass", "area", "conn_PSMC3", "path_OXPHOS_ETC", "path_PROTEOSTASIS_AUTOPHAGY"]
    md = f"""# B+A Cross-modal MoA Story

## Core Claim

The EE siRNA screen does not collapse into a single hit/no-hit ranking. It resolves into cross-modal mitochondrial MoA states that combine TMRM/MitoTracker phenotypes, transcriptomic reference connectivity, pathway programs and imaging-derived model behavior.

## Cast Of States

{state_counts.to_markdown()}

## Story Arc

### 1. First axis: mitochondrial potential collapse is a reproducible phenotype, but not always toxicity

The uncoupling states show strong negative per-mito dPsi and TMRM area relative to neutral targets. Mixed uncoupling/biogenesis targets add high mito mass on top of the collapse phenotype, which explains why a simple TMRM high/low view is not enough.

Key tests:

{format_test_table(tests, 'uncoupling_states_vs_neutral')}

### 2. Second axis: toxic collapse is a separable MoA confounder

Toxic-collapse targets are not merely strong uncouplers. They show high toxicity, deeper TMRM-area collapse, stronger per-mito dPsi loss and a statistically higher apoptosis/toxicity program. PSMC3 connectivity and proteostasis/autophagy are numerically elevated but are not significant in this small toxic-collapse group, so they remain directional supporting evidence rather than a primary claim.

Key tests:

{format_test_table(tests, 'toxic_collapse_vs_all_other_states')}

### 3. Third axis: biogenesis-like imaging states are not the canonical PGC1A/TFAM story

Biogenesis-like targets have high MitoTracker mass, but the curated mitochondrial-biogenesis transcript score does not rise in parallel. This is an important negative result: the imaging state may reflect mitochondrial abundance, morphology, dye handling or noncanonical remodeling rather than a classical biogenesis transcriptional program.

Key tests:

{format_test_table(tests, 'biogenesis_like_vs_neutral')}

### 4. Cross-modal lesson: C24 is a phenotype upper bound; expression + BF is a MoA lens

C24 features track mitochondrial dye phenotypes strongly, so they are useful as an upper bound. The more interesting MoA story comes from where expression/BF and transcriptomic programs agree or disagree with C24-driven phenotype states.

## Representative Targets

{reps[rep_cols].round(3).to_markdown(index=False)}

## Volcano Highlights

{top_genes[['contrast', 'symbol', 'effect_ntc_z', 'fdr', 'curated_pathway_gene']].round(4).to_markdown(index=False)}

## Main Figures

- `figs/story_state_effect_boxplots.png` - state-level statistical evidence.
- `figs/story_representative_target_evidence_strips.png` - representative target evidence strips.
- `figs/crossmodal_moa_map.png` - integrated MoA map.
- `figs/state_moa_summary_heatmap.png` - state-level MoA summary.
- `figs/transcriptomic_state_volcano.png` - state-vs-neutral transcriptomic volcano.

## Working Title

Cross-modal mitochondrial MoA states in an EE perturbation screen

## Next Figure-Level Tasks

1. Polish the MoA map labels into a clean main figure.
2. Turn the state boxplots into a compact panel with FDR callouts.
3. Use the representative evidence strip as the bridge from global states to named targets.
4. Move prioritization to a supplement; keep the main text focused on mechanism-state discovery.
"""
    (out_dir / "B_A_crossmodal_moa_story.md").write_text(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--volcano", default="output/2026-06-22/ba_multimodal_plan/transcriptomic_state_volcano.csv")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.table)
    volcano = pd.read_csv(args.volcano) if Path(args.volcano).exists() else pd.DataFrame()
    tests = run_story_tests(table)
    reps = select_representatives(table)
    rep_table = plot_evidence_strips(reps, figs / "story_representative_target_evidence_strips.png")
    plot_state_boxplots(table, figs / "story_state_effect_boxplots.png")
    top_genes = top_volcano_genes(volcano)

    tests.to_csv(out_dir / "state_moa_tests.csv", index=False)
    rep_table.to_csv(out_dir / "representative_target_evidence.csv", index=False)
    top_genes.to_csv(out_dir / "story_volcano_highlight_genes.csv", index=False)
    write_story(table, tests, reps, top_genes, out_dir)

    print(f"[ba_moa_story] wrote {out_dir / 'state_moa_tests.csv'}")
    print(f"[ba_moa_story] wrote {out_dir / 'representative_target_evidence.csv'}")
    print(f"[ba_moa_story] wrote {out_dir / 'story_volcano_highlight_genes.csv'}")
    print(f"[ba_moa_story] wrote {out_dir / 'B_A_crossmodal_moa_story.md'}")
    print(f"[ba_moa_story] figures under {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())