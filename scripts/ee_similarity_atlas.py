#!/usr/bin/env python
"""Compare target-target similarity atlases across DRUG-seq and DINOv2 imaging.

This script asks whether target similarity relationships are preserved between
the transcriptomic perturbation space and the image-feature spaces. It produces
pairwise similarity matrices, matrix-agreement statistics and neighbor-overlap
summaries.

Run with the scvi/anndata environment:

    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/ee_similarity_atlas.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

STATE_ORDER = [
    "toxic_collapse",
    "mixed_uncoupling_biogenesis",
    "uncoupler_like",
    "biogenesis_like",
    "neutral_or_uncertain",
]
STATE_LABEL = {
    "toxic_collapse": "Toxic",
    "mixed_uncoupling_biogenesis": "Mixed",
    "uncoupler_like": "Uncoupler",
    "biogenesis_like": "Bio-like",
    "neutral_or_uncertain": "Neutral",
}
SPACE_CONFIG = {
    "drugseq": {"label": "DRUG-seq expression delta", "kind": "expression", "obsm": "X_zscore_hvg"},
    "dino_c1": {"label": "DINOv2 C1 brightfield", "kind": "dino", "obsm": "X_dino_c1"},
    "dino_c24": {"label": "DINOv2 C24 mitochondrial", "kind": "dino", "obsm": "X_dino_c24"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--out", default="output/2026-06-23/ee_drugseq_dino_similarity")
    parser.add_argument("--min-target-wells", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clip-dino-z", type=float, default=8.0)
    return parser.parse_args()


def as_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)


def batch_ntc_z(X: np.ndarray, obs: pd.DataFrame, clip: float) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=np.float32)
    groups = obs["group"].astype(str).to_numpy()
    plates = obs["plate"].astype(str).to_numpy()
    for plate in pd.unique(plates):
        mask = plates == plate
        ntc = mask & (groups == "NTC")
        ref = X[ntc] if ntc.sum() >= 2 else X[mask]
        mu = np.nanmean(ref, axis=0)
        sd = np.nanstd(ref, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        out[mask] = (X[mask] - mu) / sd
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def robust_standardize_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def build_target_meta(obs: pd.DataFrame, table: pd.DataFrame, min_target_wells: int) -> pd.DataFrame:
    rows = []
    obs = obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["category"] = obs["category"].astype(str)
    for group, idx in obs.groupby("group", sort=True).indices.items():
        sub = obs.iloc[np.asarray(idx)]
        category = str(sub["category"].mode().iloc[0])
        if category != "Target" or len(idx) < min_target_wells:
            continue
        rows.append({
            "group": group,
            "category": category,
            "n_wells": int(len(idx)),
            "tox_rate_well": float(sub["tox_flag"].astype(bool).mean()) if "tox_flag" in sub else np.nan,
        })
    meta = pd.DataFrame(rows)
    extra = table[["group", "state_class", "recommendation", "tox_rate", "kd_tier"]].drop_duplicates("group")
    meta = meta.merge(extra, on="group", how="left")
    meta["state_sort"] = meta["state_class"].map({s: i for i, s in enumerate(STATE_ORDER)}).fillna(len(STATE_ORDER)).astype(int)
    return meta.sort_values(["state_sort", "group"]).reset_index(drop=True)


def target_centroid_matrix(X: np.ndarray, obs: pd.DataFrame, groups: list[str]) -> np.ndarray:
    obs_groups = obs["group"].astype(str).to_numpy()
    rows = []
    for group in groups:
        mask = obs_groups == group
        rows.append(X[mask].mean(axis=0))
    return np.vstack(rows)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = robust_standardize_features(X)
    norms = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norms[norms < 1e-12] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    return np.clip(sim, -1.0, 1.0)


def pearson_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = robust_standardize_features(X)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    norms[norms < 1e-12] = 1.0
    sim = (X / norms) @ (X / norms).T
    return np.clip(sim, -1.0, 1.0)


def upper_triangle_values(M: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices_from(M, k=1)
    return M[i, j]


def compare_similarity_matrices(mats: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    keys = list(mats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = upper_triangle_values(mats[a])
            vb = upper_triangle_values(mats[b])
            rows.append({
                "space_a": a,
                "space_b": b,
                "n_pairs": int(len(va)),
                "pearson_upper_triangle": pearsonr(va, vb)[0],
                "spearman_upper_triangle": spearmanr(va, vb)[0],
                "mean_abs_similarity_delta": float(np.mean(np.abs(va - vb))),
            })
    return pd.DataFrame(rows)


def neighbor_overlap(mats: dict[str, np.ndarray], groups: list[str], top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    neighbor_sets: dict[str, list[set[int]]] = {}
    for key, mat in mats.items():
        order = np.argsort(-mat, axis=1)
        sets = []
        for row_i, row_order in enumerate(order):
            keep = [int(x) for x in row_order if int(x) != row_i][:top_k]
            sets.append(set(keep))
        neighbor_sets[key] = sets

    summary_rows = []
    detail_rows = []
    keys = list(mats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlaps = []
            jaccards = []
            for target_i, group in enumerate(groups):
                sa = neighbor_sets[a][target_i]
                sb = neighbor_sets[b][target_i]
                inter = len(sa & sb)
                union = len(sa | sb)
                overlaps.append(inter / top_k)
                jaccards.append(inter / union if union else np.nan)
                detail_rows.append({
                    "space_a": a,
                    "space_b": b,
                    "group": group,
                    "top_k": top_k,
                    "overlap_fraction": inter / top_k,
                    "jaccard": inter / union if union else np.nan,
                    "shared_neighbors": ";".join(groups[idx] for idx in sorted(sa & sb)),
                })
            summary_rows.append({
                "space_a": a,
                "space_b": b,
                "top_k": top_k,
                "mean_overlap_fraction": float(np.nanmean(overlaps)),
                "mean_jaccard": float(np.nanmean(jaccards)),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def state_block_similarity(M: np.ndarray, meta: pd.DataFrame, space: str) -> pd.DataFrame:
    rows = []
    states = [s for s in STATE_ORDER if s in set(meta["state_class"])]
    state_to_idx = {s: np.where(meta["state_class"].to_numpy() == s)[0] for s in states}
    for state_a in states:
        for state_b in states:
            ia = state_to_idx[state_a]
            ib = state_to_idx[state_b]
            block = M[np.ix_(ia, ib)]
            if state_a == state_b:
                tri = block[np.triu_indices_from(block, k=1)]
                mean_sim = float(np.nanmean(tri)) if len(tri) else np.nan
            else:
                mean_sim = float(np.nanmean(block))
            rows.append({
                "space": space,
                "state_a": state_a,
                "state_b": state_b,
                "mean_similarity": mean_sim,
                "n_a": int(len(ia)),
                "n_b": int(len(ib)),
            })
    return pd.DataFrame(rows)


def plot_matrix(M: np.ndarray, meta: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    starts = []
    labels = []
    cursor = 0
    for state in STATE_ORDER:
        n = int(meta["state_class"].eq(state).sum())
        if n == 0:
            continue
        starts.append(cursor)
        labels.append(STATE_LABEL.get(state, state))
        cursor += n
        ax.axhline(cursor - 0.5, color="black", lw=0.5)
        ax.axvline(cursor - 0.5, color="black", lw=0.5)
    mids = []
    cursor = 0
    for state in STATE_ORDER:
        n = int(meta["state_class"].eq(state).sum())
        if n:
            mids.append(cursor + (n - 1) / 2)
            cursor += n
    ax.set_xticks(mids)
    ax.set_yticks(mids)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="similarity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_scatter(mats: dict[str, np.ndarray], out: Path) -> None:
    pairs = [("drugseq", "dino_c1"), ("drugseq", "dino_c24"), ("dino_c1", "dino_c24")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (a, b) in zip(axes, pairs):
        va = upper_triangle_values(mats[a])
        vb = upper_triangle_values(mats[b])
        ax.hexbin(va, vb, gridsize=45, cmap="Blues", mincnt=1)
        r = pearsonr(va, vb)[0]
        rho = spearmanr(va, vb)[0]
        ax.set_xlabel(f"{a} similarity")
        ax.set_ylabel(f"{b} similarity")
        ax.set_title(f"{a} vs {b}\nr={r:.2f}, rho={rho:.2f}")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_block_heatmap(blocks: pd.DataFrame, out: Path) -> None:
    spaces = list(pd.unique(blocks["space"]))
    fig, axes = plt.subplots(1, len(spaces), figsize=(5.4 * len(spaces), 4.8), squeeze=False)
    for ax, space in zip(axes.ravel(), spaces):
        sub = blocks[blocks["space"].eq(space)]
        mat = sub.pivot(index="state_a", columns="state_b", values="mean_similarity").reindex(index=STATE_ORDER, columns=STATE_ORDER)
        im = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        labels = [STATE_LABEL.get(s, s) for s in STATE_ORDER]
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_yticklabels(labels)
        ax.set_title(space)
        for i in range(len(STATE_ORDER)):
            for j in range(len(STATE_ORDER)):
                val = mat.iloc[i, j]
                ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return view.to_html(index=False, escape=True, classes="data")


def render_html(out: Path, compare: pd.DataFrame, nn_summary: pd.DataFrame, blocks: pd.DataFrame) -> None:
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DRUG-seq vs DINOv2 Similarity Atlas</title>
  <style>
    body {{ margin:0; background:#f6f8fa; color:#17212b; font-family:"Aptos","Segoe UI",sans-serif; line-height:1.5; }}
    main {{ width:min(1180px, calc(100vw - 36px)); margin:24px auto 46px; }}
    header, section {{ background:white; border:1px solid #d7dee6; border-radius:8px; padding:22px; margin-bottom:16px; box-shadow:0 8px 20px rgba(23,33,43,.05); }}
    h1 {{ margin:0 0 10px; font-size:32px; }} h2 {{ color:#0b4f7a; margin:0 0 12px; }}
    .lead {{ font-size:18px; max-width:86ch; color:#263747; }} .muted {{ color:#5e6a75; }}
    table.data {{ border-collapse:collapse; width:100%; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid #d7dee6; padding:7px 8px; text-align:left; }} table.data th {{ background:#eef4f8; color:#0b4f7a; }}
    img {{ max-width:100%; border:1px solid #d7dee6; border-radius:8px; background:white; }}
    code {{ background:#f0f3f6; padding:1px 4px; border-radius:4px; }}
  </style>
</head>
<body><main>
  <header>
    <p class="muted">Generated 2026-06-23</p>
    <h1>DRUG-seq vs DINOv2 Similarity Atlas</h1>
    <p class="lead">Target-target similarity matrices compare whether transcriptomic perturbation neighborhoods are preserved in C1 brightfield and C24 mitochondrial DINOv2 image-feature spaces.</p>
  </header>
  <section>
    <h2>Matrix Agreement</h2>
    <p>Upper-triangle correlations compare all target-target similarities. Positive values mean two spaces organize target pairs similarly.</p>
    {html_table(compare)}
    <p class="muted">Interpretation: this is a global matrix comparison, stricter than t-SNE/UMAP visualization.</p>
  </section>
  <section>
    <h2>Similarity Scatter</h2>
    <img src="figs/similarity_upper_triangle_scatter.png" alt="similarity scatter">
  </section>
  <section>
    <h2>Nearest-neighbor Overlap</h2>
    <p>Mean overlap of top-k neighbors per target. This answers whether each target's closest neighbors are similar across spaces.</p>
    {html_table(nn_summary)}
  </section>
  <section>
    <h2>Similarity Matrices</h2>
    <img src="figs/drugseq_similarity_matrix.png" alt="drugseq similarity matrix">
    <img src="figs/dino_c1_similarity_matrix.png" alt="dino c1 similarity matrix">
    <img src="figs/dino_c24_similarity_matrix.png" alt="dino c24 similarity matrix">
  </section>
  <section>
    <h2>State-level Block Similarity</h2>
    <img src="figs/state_block_similarity_heatmap.png" alt="state block similarity heatmap">
    {html_table(blocks, max_rows=50)}
  </section>
  <section>
    <h2>Files</h2>
    <ul>
      <li><code>similarity_matrix_agreement.csv</code></li>
      <li><code>neighbor_overlap_summary.csv</code></li>
      <li><code>neighbor_overlap_by_target.csv</code></li>
      <li><code>state_block_similarity.csv</code></li>
      <li><code>*_similarity_matrix.csv</code></li>
    </ul>
  </section>
</main></body></html>
"""
    (out / "similarity_atlas_report.html").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    import anndata as ad

    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.adata)
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["plate"] = obs["plate"].astype(str)
    table = pd.read_csv(args.table)
    meta = build_target_meta(obs, table, args.min_target_wells)
    groups = meta["group"].tolist()

    matrices = {}
    blocks = []
    for space, cfg in SPACE_CONFIG.items():
        X = as_dense(adata.obsm[cfg["obsm"]]).astype(np.float32)
        if cfg["kind"] == "dino":
            X = batch_ntc_z(X, obs, args.clip_dino_z)
        centroids = target_centroid_matrix(X, obs, groups)
        sim = pearson_similarity_matrix(centroids) if cfg["kind"] == "expression" else cosine_similarity_matrix(centroids)
        matrices[space] = sim
        pd.DataFrame(sim, index=groups, columns=groups).to_csv(out / f"{space}_similarity_matrix.csv")
        plot_matrix(sim, meta, f"{cfg['label']} target similarity", figs / f"{space}_similarity_matrix.png")
        blocks.append(state_block_similarity(sim, meta, space))

    compare = compare_similarity_matrices(matrices)
    nn_summary, nn_detail = neighbor_overlap(matrices, groups, args.top_k)
    block_df = pd.concat(blocks, ignore_index=True)
    compare.to_csv(out / "similarity_matrix_agreement.csv", index=False)
    nn_summary.to_csv(out / "neighbor_overlap_summary.csv", index=False)
    nn_detail.to_csv(out / "neighbor_overlap_by_target.csv", index=False)
    block_df.to_csv(out / "state_block_similarity.csv", index=False)
    meta.to_csv(out / "similarity_target_metadata.csv", index=False)
    plot_scatter(matrices, figs / "similarity_upper_triangle_scatter.png")
    plot_block_heatmap(block_df, figs / "state_block_similarity_heatmap.png")
    render_html(out, compare, nn_summary, block_df)

    manifest = {
        "n_targets": int(len(groups)),
        "spaces": list(SPACE_CONFIG),
        "top_k": args.top_k,
        "outputs": [
            "similarity_atlas_report.html",
            "similarity_matrix_agreement.csv",
            "neighbor_overlap_summary.csv",
            "neighbor_overlap_by_target.csv",
            "state_block_similarity.csv",
        ],
    }
    (out / "similarity_atlas_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[similarity_atlas] wrote {out / 'similarity_atlas_report.html'}")
    print(compare.to_string(index=False))
    print(nn_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())