#!/usr/bin/env python
"""Implement the EE DRUG-seq next-step plan and summarize it as HTML.

This script turns the Jira follow-up plan into concrete, versioned outputs:

1. a 10-15 target wet-lab shortlist;
2. a transcriptomic-only MoA scoring layer;
3. a toxic-sentinel action table;
4. benchmark acceptance gates for future virtual-cell models;
5. a self-contained HTML report plus zip bundle.

Run:

    python scripts/ee_drugseq_next_steps.py
"""
from __future__ import annotations

import argparse
import html
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


STATE_ORDER = [
    "neutral_or_uncertain",
    "uncoupler_like",
    "mixed_uncoupling_biogenesis",
    "biogenesis_like",
    "toxic_collapse",
]

STATE_LABELS = {
    "neutral_or_uncertain": "Neutral / uncertain",
    "uncoupler_like": "Uncoupler-like",
    "mixed_uncoupling_biogenesis": "Mixed uncoupling + biogenesis",
    "biogenesis_like": "Biogenesis-like",
    "toxic_collapse": "Toxic collapse",
}

STATE_COLORS = {
    "neutral_or_uncertain": "#8b9097",
    "uncoupler_like": "#1f77b4",
    "mixed_uncoupling_biogenesis": "#2ca25f",
    "biogenesis_like": "#d98c21",
    "toxic_collapse": "#c43b3b",
}

TX_SCORE_COLS = [
    "tx_uncoupler_score",
    "tx_biogenesis_score",
    "tx_stress_toxicity_score",
]

TX_LABELS = {
    "tx_uncoupler_score": "Tx uncoupler-like",
    "tx_biogenesis_score": "Tx biogenesis-like",
    "tx_stress_toxicity_score": "Tx stress/toxicity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--safety-status", default="output/2026-06-22/ba_multimodal_plan/EE_strict_tox_safety_report_status.md")
    parser.add_argument("--benchmark-scores", default="output/2026-06-22/vcell_benchmark_research/expression_delta_benchmark_scores.csv")
    parser.add_argument("--similarity-agreement", default="output/2026-06-23/ee_drugseq_dino_similarity/similarity_matrix_agreement.csv")
    parser.add_argument("--drugseq-report", default="output/2026-06-23/drugseq_report/EE_DRUG_seq_dedicated_analysis_report.html")
    parser.add_argument("--similarity-report", default="output/2026-06-23/ee_drugseq_dino_similarity/similarity_atlas_report.html")
    parser.add_argument("--benchmark-report", default="output/2026-06-22/vcell_benchmark_research/EE_DrugSeq_v1_benchmark_report.html")
    parser.add_argument("--out", default="output/2026-06-23/drugseq_next_steps")
    parser.add_argument("--shortlist-size", type=int, default=15)
    return parser.parse_args()


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad < 1e-9:
        scale = numeric.std(ddof=0)
    else:
        scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return ((numeric - median) / scale).clip(-3, 3).fillna(0.0)


def html_table(df: pd.DataFrame, *, max_rows: int = 30, float_format: str = "{:.3f}") -> str:
    if df.empty:
        return "<p class='muted'>No data available.</p>"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else float_format.format(float(value)))
    return view.to_html(index=False, escape=True, classes="data")


def safe_relative(path: Path, out: Path) -> str:
    return html.escape(path.relative_to(out).as_posix())


def broad_imaging_bucket(state: str) -> str:
    if state in {"uncoupler_like", "mixed_uncoupling_biogenesis"}:
        return "uncoupling_axis"
    if state == "biogenesis_like":
        return "biogenesis_axis"
    if state == "toxic_collapse":
        return "stress_toxicity_axis"
    return "neutral_or_uncertain"


def compute_transcriptomic_moa(targets: pd.DataFrame) -> pd.DataFrame:
    """Build a transparent transcriptomic-only MoA scoring layer.

    Only DRUG-seq-derived columns are used: expression-signature connectivity
    to reference perturbations plus marker-gene pathway scores. Imaging
    phenotype columns are added back only for auditing agreement.
    """
    out = targets.copy()
    required = [
        "conn_BAM15", "conn_MK8722", "conn_PSMC3",
        "conn_BAM15_minus_MK8722", "conn_MK8722_minus_BAM15", "conn_toxicity_margin",
        "path_OXPHOS_ETC", "path_APOPTOSIS_TOXICITY", "path_ISR_ER_STRESS", "path_PROTEOSTASIS_AUTOPHAGY",
    ]
    for col in required:
        if col not in out:
            out[col] = 0.0
        out[f"rz_{col}"] = robust_z(out[col])

    out["tx_uncoupler_score"] = (
        0.45 * out["rz_conn_BAM15_minus_MK8722"]
        + 0.35 * out["rz_conn_BAM15"]
        + 0.20 * out["rz_path_OXPHOS_ETC"]
    )
    out["tx_biogenesis_score"] = (
        0.55 * out["rz_conn_MK8722_minus_BAM15"]
        + 0.35 * out["rz_conn_MK8722"]
        - 0.10 * out["rz_path_APOPTOSIS_TOXICITY"]
    )
    out["tx_stress_toxicity_score"] = (
        0.30 * out["rz_conn_toxicity_margin"]
        + 0.25 * out["rz_conn_PSMC3"]
        + 0.20 * out["rz_path_APOPTOSIS_TOXICITY"]
        + 0.15 * out["rz_path_ISR_ER_STRESS"]
        + 0.10 * out["rz_path_PROTEOSTASIS_AUTOPHAGY"]
    )

    score_matrix = out[TX_SCORE_COLS].to_numpy(dtype=float)
    best_idx = np.nanargmax(score_matrix, axis=1)
    sorted_scores = np.sort(score_matrix, axis=1)
    best_score = sorted_scores[:, -1]
    runner_up = sorted_scores[:, -2]
    calls = np.asarray(["tx_uncoupler_like", "tx_biogenesis_like", "tx_stress_toxicity"]) [best_idx]
    uncertain = (best_score < 0.45) | ((best_score - runner_up) < 0.15)
    calls = np.where(uncertain, "tx_neutral_or_uncertain", calls)

    out["tx_primary_call"] = calls
    out["tx_primary_score"] = best_score
    out["tx_call_margin"] = best_score - runner_up
    out["imaging_bucket"] = out["state_class"].map(broad_imaging_bucket)
    expected = {
        "tx_uncoupler_like": "uncoupling_axis",
        "tx_biogenesis_like": "biogenesis_axis",
        "tx_stress_toxicity": "stress_toxicity_axis",
        "tx_neutral_or_uncertain": "neutral_or_uncertain",
    }
    out["tx_imaging_bucket_match"] = out["tx_primary_call"].map(expected).eq(out["imaging_bucket"])
    return out


def select_shortlist(tx: pd.DataFrame, shortlist_size: int) -> pd.DataFrame:
    targets = tx[tx["category"].eq("Target")].copy()
    targets["tx_support_bonus"] = np.where(targets["tx_imaging_bucket_match"], 1.0, 0.0)
    targets["kd_bonus"] = targets["kd_tier"].map({"strong": 1.0, "weak": 0.35}).fillna(0.0)
    targets["clean_bonus"] = np.where(targets["tox_rate"].fillna(0) <= 0.0, 0.5, 0.0)
    targets["shortlist_score"] = (
        targets["consensus_score"].fillna(0)
        + targets["tx_support_bonus"]
        + targets["kd_bonus"]
        + targets["clean_bonus"]
    )
    targets["shortlist_reason"] = np.select(
        [
            targets["tx_imaging_bucket_match"].eq(True) & targets["tox_rate"].fillna(0).eq(0),
            targets["tx_imaging_bucket_match"].eq(True),
            targets["tox_rate"].fillna(0).eq(0),
        ],
        [
            "clean imaging hit with transcriptomic support",
            "imaging hit with transcriptomic support",
            "clean high-consensus imaging hit",
        ],
        default="high-consensus follow-up",
    )
    pool = targets[
        targets["recommendation"].eq("tier1_immediate_validation")
        & targets["kd_tier"].isin(["strong", "weak"])
        & targets["tox_rate"].fillna(1).le(0.05)
        & targets["qc_fail_rate"].fillna(1).le(0.25)
        & targets["n_clean_wells"].fillna(0).ge(4)
        & targets["dominant_ci_excludes_zero"].astype(bool)
    ].copy()
    pool = pool.sort_values(["shortlist_score", "consensus_score"], ascending=False)

    quotas = {
        "mixed_uncoupling_biogenesis": 6,
        "uncoupler_like": 5,
        "biogenesis_like": 4,
    }
    selected_idx: list[int] = []
    for state, quota in quotas.items():
        selected_idx.extend(pool[pool["state_class"].eq(state)].head(quota).index.tolist())
    if len(selected_idx) < shortlist_size:
        selected_idx.extend(pool.drop(index=selected_idx, errors="ignore").head(shortlist_size - len(selected_idx)).index.tolist())
    selected = pool.loc[selected_idx].drop_duplicates("group").head(shortlist_size).copy()
    selected["rank"] = np.arange(1, len(selected) + 1)
    keep = [
        "rank", "group", "state_class", "kd_tier", "shortlist_score", "consensus_score",
        "permito", "mitomass", "area", "tox_rate", "tx_primary_call", "tx_primary_score",
        "tx_call_margin", "tx_imaging_bucket_match", "shortlist_reason",
    ]
    return selected[keep]


def parse_safety_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["group", "new_safety_result", "risk_score", "matched_keywords"])
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(parts)
    if len(rows) < 2:
        return pd.DataFrame(columns=["group", "new_safety_result", "risk_score", "matched_keywords"])
    header, data = rows[0], rows[1:]
    table = pd.DataFrame(data, columns=header)
    rename = {
        "Target": "group",
        "New dedicated safety result": "new_safety_result",
        "Risk score": "risk_score",
        "Matched keywords": "matched_keywords",
        "Pre-existing safety status": "pre_existing_safety_status",
        "vCell tox_rate": "safety_vcell_tox_rate",
    }
    table = table.rename(columns=rename)
    if "risk_score" in table:
        table["risk_score"] = pd.to_numeric(table["risk_score"], errors="coerce")
    return table


def build_toxic_sentinel(tx: pd.DataFrame, safety: pd.DataFrame) -> pd.DataFrame:
    tox = tx[tx["state_class"].eq("toxic_collapse")].copy()
    cols = [
        "group", "kd_tier", "tox_rate", "permito", "area", "tx_stress_toxicity_score",
        "path_APOPTOSIS_TOXICITY", "path_ISR_ER_STRESS", "path_PROTEOSTASIS_AUTOPHAGY",
    ]
    tox = tox[cols]
    tox = tox.merge(safety, on="group", how="left")
    tox["sentinel_action"] = np.select(
        [
            tox["new_safety_result"].eq("RED"),
            tox["new_safety_result"].eq("YELLOW"),
        ],
        [
            "exclude from clean EE shortlist; track as safety/essentiality sentinel",
            "orthogonal repeat or literature check before any rescue",
        ],
        default="manual safety review",
    )
    tox["next_experiment"] = np.select(
        [
            tox["group"].eq("PSMC3"),
            tox["kd_tier"].eq("failed"),
            tox["new_safety_result"].eq("RED"),
        ],
        [
            "use as positive toxic-collapse sentinel and safety concordance anchor",
            "repeat KD / exclude off-target before interpretation",
            "do not advance as clean hit; document liability mechanism",
        ],
        default="repeat cell-count/TMRM check plus transcriptomic stress readout",
    )
    return tox.sort_values(["tox_rate", "risk_score"], ascending=[False, False])


def build_benchmark_gates(scores: pd.DataFrame) -> pd.DataFrame:
    score = scores.set_index("model")
    c24 = score.loc["c24_nearest_delta"]
    c1 = score.loc["c1_nearest_delta"]
    loo = score.loc["loo_mean_delta"]
    zero = score.loc["zero_delta_ntc"]
    rows = [
        {
            "gate": "minimum_nonleaky_expression_model",
            "pds_threshold": max(float(zero["pds"]), float(loo["pds"])) + 0.05,
            "des_topk_threshold": float(loo["des_topk"]),
            "meaning": "must beat no-effect / leave-one-out mean baselines under leave-target-out evaluation",
        },
        {
            "gate": "strong_crossmodal_baseline",
            "pds_threshold": float(c1["pds"]),
            "des_topk_threshold": float(c1["des_topk"]),
            "meaning": "must beat brightfield-nearest sanity baseline",
        },
        {
            "gate": "stretch_mitochondrial_baseline",
            "pds_threshold": float(c24["pds"]),
            "des_topk_threshold": float(c24["des_topk"]),
            "meaning": "stretch target: beat mitochondrial C24-nearest upper sanity comparator",
        },
    ]
    return pd.DataFrame(rows)


def plot_shortlist(shortlist: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    plot_df = shortlist.sort_values("shortlist_score", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    colors = [STATE_COLORS.get(state, "#777777") for state in plot_df["state_class"]]
    ax.barh(plot_df["group"], plot_df["shortlist_score"], color=colors)
    for _, row in plot_df.iterrows():
        ax.text(row["shortlist_score"] + 0.05, row["group"], f"{row['kd_tier']} | {row['tx_primary_call'].replace('tx_', '')}", va="center", fontsize=8)
    ax.set_xlabel("shortlist score")
    ax.set_title("Wet-lab shortlist: balanced high-confidence candidates", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    path = figs / "shortlist_priority.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_tx_scores(tx: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    target = tx[tx["category"].eq("Target")].copy()
    top = target.sort_values("tx_primary_score", ascending=False).head(50)
    matrix = top[TX_SCORE_COLS].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 11.2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["group"], fontsize=7)
    ax.set_xticks(np.arange(len(TX_SCORE_COLS)))
    ax.set_xticklabels([TX_LABELS[col] for col in TX_SCORE_COLS], rotation=25, ha="right")
    ax.set_title("Transcriptomic-only MoA scores: top 50 strongest calls", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("robust expression-score units")
    fig.tight_layout()
    path = figs / "transcriptomic_moa_scores_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_tx_agreement(tx: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    target = tx[tx["category"].eq("Target")].copy()
    calls = ["tx_uncoupler_like", "tx_biogenesis_like", "tx_stress_toxicity", "tx_neutral_or_uncertain"]
    buckets = ["uncoupling_axis", "biogenesis_axis", "stress_toxicity_axis", "neutral_or_uncertain"]
    tab = pd.crosstab(target["tx_primary_call"], target["imaging_bucket"]).reindex(index=calls, columns=buckets).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    im = ax.imshow(tab.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(tab.columns)))
    ax.set_xticklabels(tab.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(tab.index)))
    ax.set_yticklabels([idx.replace("tx_", "") for idx in tab.index])
    for y in range(tab.shape[0]):
        for x in range(tab.shape[1]):
            ax.text(x, y, str(int(tab.iloc[y, x])), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="# targets")
    ax.set_title("Transcriptomic-only call vs imaging-derived state bucket", fontweight="bold")
    fig.tight_layout()
    path = figs / "transcriptomic_vs_imaging_agreement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_toxic_sentinel(sentinel: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    colors = sentinel["new_safety_result"].map({"RED": "#b83c3c", "YELLOW": "#c9952d"}).fillna("#777777")
    ax.scatter(sentinel["tox_rate"], sentinel["tx_stress_toxicity_score"], s=180, c=colors, edgecolor="white", linewidth=0.8)
    for _, row in sentinel.iterrows():
        ax.text(row["tox_rate"] + 0.01, row["tx_stress_toxicity_score"], row["group"], fontsize=9, va="center")
    ax.axvline(0.5, color="#b83c3c", linestyle="--", lw=1, label="hard tox gate")
    ax.axvline(0.2, color="#c9952d", linestyle="--", lw=1, label="soft tox gate")
    ax.set_xlabel("tox_rate from imaging cell-count loss")
    ax.set_ylabel("transcriptomic stress/toxicity score")
    ax.set_title("Toxic-collapse sentinel targets", fontweight="bold")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figs / "toxic_sentinel_plan.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_benchmark_gates(scores: pd.DataFrame, gates: pd.DataFrame, out: Path) -> Path:
    figs = out / "figs"
    plot_df = scores.sort_values("pds", ascending=True)
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.barh(plot_df["model"], plot_df["pds"], color="#2c628f")
    for _, row in gates.iterrows():
        ax.axvline(row["pds_threshold"], linestyle="--", lw=1.2, label=row["gate"])
    ax.set_xlabel("PDS (higher is better)")
    ax.set_title("EE-DrugSeq-v1 benchmark gates for future virtual-cell models", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = figs / "benchmark_acceptance_gates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_plan_markdown(out: Path) -> Path:
    text = """# EE DRUG-seq Next-step Implementation Plan

## Implemented outputs

1. `wetlab_shortlist.csv`: balanced 15-target wet-lab shortlist from tier-1 candidates.
2. `transcriptomic_moa_scores.csv`: transcriptomic-only MoA scores from DRUG-seq connectivity and marker-gene programs.
3. `transcriptomic_moa_call_agreement.csv`: cross-tab between transcriptomic-only calls and imaging-derived state buckets.
4. `toxic_sentinel_plan.csv`: toxic-collapse targets with safety status and next action.
5. `benchmark_acceptance_gates.csv`: minimum / strong / stretch gates for future virtual-cell models.

## Decision principles

- The shortlist still uses imaging state and KD quality because it is a wet-lab validation list.
- The transcriptomic-only MoA score deliberately does not use `permito`, `mitomass`, `area`, `tox_rate`, or `qc_fail_rate`.
- Toxic-collapse targets are handled as safety/essentiality sentinels, not clean EE MoA hits.
- Benchmark gates are leave-target-out gates; random splits are leakage audits only.
"""
    path = out / "next_step_plan.md"
    path.write_text(text, encoding="utf-8")
    return path


def render_html(
    out: Path,
    figures: dict[str, Path],
    shortlist: pd.DataFrame,
    tx: pd.DataFrame,
    agreement: pd.DataFrame,
    sentinel: pd.DataFrame,
    gates: pd.DataFrame,
    scores: pd.DataFrame,
    similarity: pd.DataFrame,
    source_links: dict[str, str],
) -> Path:
    n_targets = int(tx[tx["category"].eq("Target")]["group"].nunique())
    shortlist_states = shortlist["state_class"].value_counts().to_dict()
    tx_target = tx[tx["category"].eq("Target")]
    match_rate = float(tx_target["tx_imaging_bucket_match"].mean()) if len(tx_target) else float("nan")
    best_benchmark = scores.sort_values("pds", ascending=False).iloc[0]
    c24_vs_c1 = similarity.set_index(["space_a", "space_b"])
    c24_spearman = float(c24_vs_c1.loc[("drugseq", "dino_c24"), "spearman_upper_triangle"])
    c1_spearman = float(c24_vs_c1.loc[("drugseq", "dino_c1"), "spearman_upper_triangle"])

    fig = {key: safe_relative(path, out) for key, path in figures.items()}
    sources = "".join(f"<li><code>{html.escape(label)}</code>: {html.escape(path)}</li>" for label, path in source_links.items())
    state_mix = ", ".join(f"{STATE_LABELS.get(k, k)}={v}" for k, v in shortlist_states.items())

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EE DRUG-seq Next-step Implementation</title>
  <style>
    :root {{
      --ink: #132125;
      --muted: #657277;
      --line: #d8dfdc;
      --panel: #ffffff;
      --paper: #f5f1e8;
      --teal: #176d6b;
      --blue: #2c628f;
      --amber: #c9822a;
      --red: #b83c3c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif; color: var(--ink); background: linear-gradient(180deg, #f5f1e8 0%, #edf4f0 48%, #fbfcfa 100%); line-height: 1.62; }}
    header {{ padding: 46px 34px 34px; background: linear-gradient(135deg, #11343b 0%, #176d6b 56%, #d4a155 100%); color: #fff; }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 0 26px 44px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px; opacity: 0.82; }}
    h1 {{ margin: 10px 0 10px; font-size: clamp(30px, 4vw, 50px); line-height: 1.08; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    h3 {{ margin: 22px 0 10px; font-size: 18px; }}
    .subtitle {{ max-width: 900px; opacity: 0.9; font-size: 17px; }}
    .section {{ margin-top: 26px; padding: 25px; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,0.9); box-shadow: 0 14px 34px rgba(35,52,55,0.08); }}
    .grid {{ display: grid; gap: 16px; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .metric {{ padding: 15px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfa; }}
    .metric .value {{ display: block; color: var(--teal); font-weight: 800; font-size: 27px; }}
    .metric .label {{ display: block; color: var(--muted); font-size: 13px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 7px; background: #fff; }}
    figcaption {{ margin-top: 7px; color: var(--muted); font-size: 13px; }}
    table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.data th, table.data td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    table.data th {{ background: #edf3f0; color: #22373b; }}
    .table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    .callout {{ padding: 15px 18px; border-left: 5px solid var(--teal); background: #edf6f2; border-radius: 7px; }}
    .muted {{ color: var(--muted); }}
    code {{ background: #eef2ef; padding: 2px 5px; border-radius: 4px; }}
    ul {{ padding-left: 20px; }}
    @media (max-width: 860px) {{ .grid.two, .grid.three {{ grid-template-columns: 1fr; }} .wrap {{ padding: 0 14px 32px; }} .section {{ padding: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="eyebrow">EE DRUG-seq | next-step implementation</div>
      <h1>把 Jira 下一步计划落成可执行 shortlist、MoA score、sentinel 和 benchmark gates</h1>
      <p class="subtitle">本报告把前一轮 DRUG-seq/成像联合分析的计划转成可复查输出。重点是 wet-lab shortlist、transcriptomic-only MoA score、tox sentinel handling、future vCell benchmark 门槛和 versioned bundle。</p>
    </div>
  </header>
  <main class="wrap">
    <section class="section">
      <h2>Executive Summary</h2>
      <div class="callout">已实现 5 个下一步产物：15-target wet-lab shortlist、DRUG-seq-only MoA scoring、toxic sentinel plan、benchmark gates、versioned HTML bundle。transcriptomic-only score 不使用 <code>permito/mitomass/area/tox_rate</code>，因此可作为 imaging-driven state 的独立审计层。</div>
      <div class="grid three" style="margin-top: 18px;">
        <div class="metric"><span class="value">{n_targets}</span><span class="label">targets scored</span></div>
        <div class="metric"><span class="value">{len(shortlist)}</span><span class="label">wet-lab shortlist</span></div>
        <div class="metric"><span class="value">{match_rate:.1%}</span><span class="label">Tx-only vs imaging bucket match</span></div>
        <div class="metric"><span class="value">{len(sentinel)}</span><span class="label">toxic sentinels</span></div>
        <div class="metric"><span class="value">{best_benchmark['pds']:.3f}</span><span class="label">best current PDS ({best_benchmark['model']})</span></div>
        <div class="metric"><span class="value">{c24_spearman:.3f}</span><span class="label">DRUG-seq vs C24 similarity Spearman</span></div>
      </div>
    </section>

    <section class="section">
      <h2>1. Wet-lab Shortlist</h2>
      <p>选择规则：tier-1 candidates、KD strong/weak、tox_rate=0、QC可接受、clean wells >=4、dominant axis CI 排除 0；然后按 consensus、KD、tox clean 和 transcriptomic support 加权，保留 state balance。当前 shortlist state mix：{html.escape(state_mix)}。</p>
      <figure><img src="{fig['shortlist']}" alt="shortlist priority"><figcaption>15-target wet-lab shortlist，标签显示 KD tier 和 transcriptomic-only call。</figcaption></figure>
      <h3>Shortlist Table</h3>
      <div class="table-scroll">{html_table(shortlist, max_rows=20)}</div>
    </section>

    <section class="section">
      <h2>2. DRUG-seq-primary MoA Score</h2>
      <p>这里新建了一个不依赖 imaging gate 的 MoA score。它只使用 DRUG-seq expression-derived reference connectivity 和 marker-gene pathway score：BAM15/MK8722/PSMC3 connectivity、BAM15-MK8722 margin、toxicity margin，以及 apoptosis/ISR/proteostasis/OXPHOS program。</p>
      <div class="grid two">
        <figure><img src="{fig['tx_scores']}" alt="transcriptomic MoA score heatmap"><figcaption>top 50 transcriptomic-only MoA calls。红/蓝表示相对全体 target 的 robust expression-score units。</figcaption></figure>
        <figure><img src="{fig['tx_agreement']}" alt="transcriptomic imaging agreement"><figcaption>DRUG-seq-only call 与 imaging-derived state bucket 的交叉表。它是审计层，不替代 imaging state。</figcaption></figure>
      </div>
      <h3>Agreement Matrix</h3>
      <div class="table-scroll">{html_table(agreement.reset_index(), max_rows=12)}</div>
    </section>

    <section class="section">
      <h2>3. Toxic Sentinel Plan</h2>
      <p>Toxic-collapse targets 不进入 clean EE shortlist，作为 toxicity/essentiality sentinel 管理。PSMC3 保留为跨项目 concordance anchor；failed KD 的 toxic target 需要先排除 KD/off-target 不确定性。</p>
      <figure><img src="{fig['sentinel']}" alt="toxic sentinel"><figcaption>tox_rate 来自 imaging cell-count loss；stress/toxicity score 来自 DRUG-seq-only scoring。</figcaption></figure>
      <div class="table-scroll">{html_table(sentinel, max_rows=12)}</div>
    </section>

    <section class="section">
      <h2>4. Benchmark Gates</h2>
      <p>未来 virtual-cell 模型必须在 leave-target-out 条件下报告这些 gate。minimum gate 要超过无效/均值 baseline；strong gate 要超过 C1-nearest；stretch gate 要超过 C24-nearest sanity comparator。</p>
      <figure><img src="{fig['benchmark']}" alt="benchmark gates"><figcaption>当前 baseline PDS 与三个 acceptance gates。</figcaption></figure>
      <div class="grid two">
        <div class="table-scroll">{html_table(gates, max_rows=10)}</div>
        <div class="table-scroll">{html_table(scores, max_rows=10)}</div>
      </div>
    </section>

    <section class="section">
      <h2>5. Versioned Bundle</h2>
      <p>本轮输出已写入同一目录，后续 Jira/slide 可直接引用这一套 versioned files。</p>
      <ul>
        <li><code>wetlab_shortlist.csv</code></li>
        <li><code>transcriptomic_moa_scores.csv</code></li>
        <li><code>transcriptomic_moa_call_agreement.csv</code></li>
        <li><code>toxic_sentinel_plan.csv</code></li>
        <li><code>benchmark_acceptance_gates.csv</code></li>
        <li><code>EE_DRUG_seq_next_steps_implementation_report.html</code></li>
        <li><code>drugseq_next_steps_bundle.zip</code></li>
      </ul>
      <h3>Source reports</h3>
      <ul>{sources}</ul>
    </section>
  </main>
</body>
</html>
"""
    path = out / "EE_DRUG_seq_next_steps_implementation_report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def create_bundle(out: Path) -> Path:
    bundle = out / "drugseq_next_steps_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != bundle:
                zf.write(path, arcname=str(out.name / path.relative_to(out)))
    return bundle


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.target_table)
    targets = targets[targets["category"].isin(["Target", "PC", "NC"])].copy()
    scores = pd.read_csv(args.benchmark_scores)
    similarity = pd.read_csv(args.similarity_agreement)
    safety = parse_safety_status(Path(args.safety_status))

    tx = compute_transcriptomic_moa(targets)
    shortlist = select_shortlist(tx, args.shortlist_size)
    sentinel = build_toxic_sentinel(tx, safety)
    gates = build_benchmark_gates(scores)
    agreement = pd.crosstab(
        tx[tx["category"].eq("Target")]["tx_primary_call"],
        tx[tx["category"].eq("Target")]["imaging_bucket"],
    )

    shortlist.to_csv(out / "wetlab_shortlist.csv", index=False)
    tx_cols = [
        "group", "category", "state_class", "imaging_bucket", "kd_tier", "tox_rate",
        "tx_uncoupler_score", "tx_biogenesis_score", "tx_stress_toxicity_score",
        "tx_primary_call", "tx_primary_score", "tx_call_margin", "tx_imaging_bucket_match",
        "conn_BAM15", "conn_MK8722", "conn_PSMC3", "conn_toxicity_margin",
        "path_OXPHOS_ETC", "path_APOPTOSIS_TOXICITY", "path_ISR_ER_STRESS", "path_PROTEOSTASIS_AUTOPHAGY",
    ]
    tx[[col for col in tx_cols if col in tx.columns]].to_csv(out / "transcriptomic_moa_scores.csv", index=False)
    agreement.to_csv(out / "transcriptomic_moa_call_agreement.csv")
    sentinel.to_csv(out / "toxic_sentinel_plan.csv", index=False)
    gates.to_csv(out / "benchmark_acceptance_gates.csv", index=False)
    write_plan_markdown(out)

    figures = {
        "shortlist": plot_shortlist(shortlist, out),
        "tx_scores": plot_tx_scores(tx, out),
        "tx_agreement": plot_tx_agreement(tx, out),
        "sentinel": plot_toxic_sentinel(sentinel, out),
        "benchmark": plot_benchmark_gates(scores, gates, out),
    }

    source_links = {
        "DRUG-seq report": args.drugseq_report,
        "similarity atlas": args.similarity_report,
        "benchmark report": args.benchmark_report,
        "target table": args.target_table,
        "safety status": args.safety_status,
    }
    report = render_html(out, figures, shortlist, tx, agreement, sentinel, gates, scores, similarity, source_links)
    bundle = create_bundle(out)

    manifest = {
        "report": report.name,
        "bundle": bundle.name,
        "tables": [
            "wetlab_shortlist.csv",
            "transcriptomic_moa_scores.csv",
            "transcriptomic_moa_call_agreement.csv",
            "toxic_sentinel_plan.csv",
            "benchmark_acceptance_gates.csv",
        ],
        "figures": {key: value.relative_to(out).as_posix() for key, value in figures.items()},
        "sources": source_links,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[drugseq_next_steps] wrote {report}")
    print(f"[drugseq_next_steps] shortlist={len(shortlist)} toxic_sentinels={len(sentinel)} bundle={bundle.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
