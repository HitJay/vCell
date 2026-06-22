#!/usr/bin/env python
"""Analyze target geometry in DINOv2 imaging feature space for the B+A story."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import pairwise_distances, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:  # noqa: SIM105 - optional dependency with a lightweight fallback flag.
    import umap  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    umap = None

STATE_ORDER = [
    "neutral_or_uncertain",
    "uncoupler_like",
    "mixed_uncoupling_biogenesis",
    "biogenesis_like",
    "toxic_collapse",
]

STATE_LABELS = {
    "neutral_or_uncertain": "Neutral",
    "uncoupler_like": "Uncoupler",
    "mixed_uncoupling_biogenesis": "Mixed",
    "biogenesis_like": "Bio-like",
    "toxic_collapse": "Toxic",
    "energizer_like": "Energizer",
}

STATE_COLORS = {
    "neutral_or_uncertain": "#a6a6a6",
    "uncoupler_like": "#e8893a",
    "mixed_uncoupling_biogenesis": "#8b88c6",
    "biogenesis_like": "#55bfa5",
    "toxic_collapse": "#c73e4d",
    "energizer_like": "#5a9f3a",
}

SELECTED_LABELS = [
    "BAM15",
    "MK8722",
    "PSMC3",
    "KANSL1",
    "RPL8",
    "RFT1",
    "CDK2AP1",
    "CHRNE",
    "DDI2",
    "TM6SF2",
    "DGAT2",
    "TAGLN",
    "G6PC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--table", default="output/2026-06-22/ba_multimodal_plan/crossmodal_moa_target_table.csv")
    parser.add_argument("--out", default="output/2026-06-22/ba_multimodal_plan")
    parser.add_argument("--clip", type=float, default=8.0, help="Clip per-plate NTC z-scored DINO features")
    parser.add_argument("--tsne-perplexity", type=float, default=18.0)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    return parser.parse_args()


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


def target_centroids(Xz: np.ndarray, obs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_cols = ["group", "category", "state_class", "recommendation", "tox_rate", "kd_tier"]
    meta = table[meta_cols].drop_duplicates("group")
    obs_small = obs[["group", "category", "tox_flag"]].copy()
    for group, idx in obs_small.groupby("group").indices.items():
        vec = Xz[np.asarray(idx)].mean(axis=0)
        sub = obs_small.iloc[np.asarray(idx)]
        rows.append(
            {
                "group": str(group),
                "category_obs": str(sub["category"].mode().iloc[0]),
                "n_wells_dino": int(len(idx)),
                "tox_rate_well": float(sub["tox_flag"].astype(bool).mean()),
                **{f"f{i:03d}": float(v) for i, v in enumerate(vec)},
            }
        )
    cent = pd.DataFrame(rows).merge(meta, on="group", how="left")
    cent["category"] = cent["category"].fillna(cent["category_obs"])
    return cent


def standardize_features(cent: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    feature_cols = [c for c in cent.columns if c.startswith("f")]
    mask = cent["category"].isin(["Target", "PC"])
    scaler = StandardScaler()
    X = scaler.fit_transform(cent.loc[mask, feature_cols].to_numpy(dtype=float))
    out = cent.loc[mask].reset_index(drop=True).copy()
    return out, X, feature_cols


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def state_distance_table(plot_df: pd.DataFrame, X: np.ndarray, channel: str) -> pd.DataFrame:
    target_mask = plot_df["category"].eq("Target") & plot_df["state_class"].isin(STATE_ORDER)
    target_df = plot_df[target_mask].reset_index(drop=True)
    target_X = X[target_mask.to_numpy()]
    centroids = {}
    for state in STATE_ORDER:
        idx = target_df["state_class"].eq(state).to_numpy()
        if idx.sum() > 0:
            centroids[state] = target_X[idx].mean(axis=0)
    rows = []
    for state_a in STATE_ORDER:
        for state_b in STATE_ORDER:
            if state_a in centroids and state_b in centroids:
                rows.append(
                    {
                        "channel": channel,
                        "state_a": state_a,
                        "state_b": state_b,
                        "rms_distance": rms_distance(centroids[state_a], centroids[state_b]),
                    }
                )
    return pd.DataFrame(rows)


def separation_summary(plot_df: pd.DataFrame, X: np.ndarray, channel: str) -> dict[str, float | str | int]:
    target_mask = plot_df["category"].eq("Target") & plot_df["state_class"].isin(STATE_ORDER)
    target_df = plot_df[target_mask].reset_index(drop=True)
    target_X = X[target_mask.to_numpy()]
    labels = target_df["state_class"].astype(str).to_numpy()
    valid_states = [s for s in STATE_ORDER if np.sum(labels == s) >= 2]
    valid = np.isin(labels, valid_states)
    sil = float(silhouette_score(target_X[valid], labels[valid])) if len(valid_states) >= 2 else float("nan")
    D = pairwise_distances(target_X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    nn_same = float(np.mean(labels[nn] == labels))
    centroids = {s: target_X[labels == s].mean(axis=0) for s in valid_states}
    within = []
    for state in valid_states:
        within.extend([rms_distance(x, centroids[state]) for x in target_X[labels == state]])
    between = []
    for i, state_a in enumerate(valid_states):
        for state_b in valid_states[i + 1 :]:
            between.append(rms_distance(centroids[state_a], centroids[state_b]))
    return {
        "channel": channel,
        "n_targets": int(len(target_df)),
        "n_states": int(len(valid_states)),
        "silhouette": sil,
        "nearest_neighbor_same_state_fraction": nn_same,
        "mean_within_state_rms": float(np.mean(within)) if within else float("nan"),
        "mean_between_state_centroid_rms": float(np.mean(between)) if between else float("nan"),
        "between_to_within_ratio": float(np.mean(between) / np.mean(within)) if within and between else float("nan"),
    }


def nearest_neighbors(plot_df: pd.DataFrame, X: np.ndarray, channel: str, selected: list[str], k: int = 8) -> pd.DataFrame:
    D = pairwise_distances(X, metric="euclidean")
    groups = plot_df["group"].astype(str).to_numpy()
    rows = []
    for group in selected:
        if group not in set(groups):
            continue
        i = int(np.where(groups == group)[0][0])
        order = np.argsort(D[i])
        rank = 0
        for j in order:
            if i == j:
                continue
            rank += 1
            rows.append(
                {
                    "channel": channel,
                    "query": group,
                    "rank": rank,
                    "neighbor": groups[j],
                    "neighbor_state": plot_df.iloc[j]["state_class"],
                    "neighbor_category": plot_df.iloc[j]["category"],
                    "euclidean_distance": float(D[i, j]),
                    "rms_distance": rms_distance(X[i], X[j]),
                }
            )
            if rank >= k:
                break
    return pd.DataFrame(rows)


def project_embedding(X: np.ndarray, method: str, args: argparse.Namespace) -> tuple[np.ndarray, str, str]:
    if method == "pca":
        pca = PCA(n_components=2, random_state=0)
        emb = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_[:2] * 100
        return emb, f"PC1 ({ev[0]:.1f}% var)", f"PC2 ({ev[1]:.1f}% var)"
    if method == "tsne":
        pca_init = PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(X)
        perplexity = min(args.tsne_perplexity, max(5.0, (X.shape[0] - 1) / 3))
        emb = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            metric="euclidean",
            random_state=0,
        ).fit_transform(pca_init)
        return emb, f"t-SNE 1 (perplexity={perplexity:g})", "t-SNE 2"
    if method == "umap":
        if umap is None:
            raise RuntimeError("umap-learn is not available in this Python environment")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=0,
        )
        emb = reducer.fit_transform(X)
        return emb, f"UMAP 1 (n_neighbors={args.umap_n_neighbors})", f"UMAP 2 (min_dist={args.umap_min_dist:g})"
    raise ValueError(f"Unknown embedding method: {method}")


def plot_state_map(plot_df: pd.DataFrame, X: np.ndarray, channel: str, method: str, out: Path, args: argparse.Namespace) -> None:
    emb, x_label, y_label = project_embedding(X, method, args)
    plot_df = plot_df.copy()
    plot_df["x"] = emb[:, 0]
    plot_df["y"] = emb[:, 1]
    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    for state in STATE_ORDER + ["energizer_like"]:
        sub = plot_df[plot_df["state_class"].eq(state)]
        if sub.empty:
            continue
        marker = "*" if (sub["category"].eq("PC")).all() else "o"
        ax.scatter(
            sub["x"],
            sub["y"],
            s=52 if marker == "o" else 160,
            alpha=0.72,
            color=STATE_COLORS.get(state, "#777777"),
            label=STATE_LABELS.get(state, state),
            edgecolor="white",
            linewidth=0.5,
            marker=marker,
        )
    target_df = plot_df[plot_df["category"].eq("Target") & plot_df["state_class"].isin(STATE_ORDER)]
    for state in STATE_ORDER:
        sub = target_df[target_df["state_class"].eq(state)]
        if sub.empty:
            continue
        centroid = sub[["x", "y"]].mean()
        ax.scatter(centroid["x"], centroid["y"], s=250, color=STATE_COLORS[state], edgecolor="black", linewidth=1.1)
        ax.annotate(
            STATE_LABELS[state],
            (centroid["x"], centroid["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    for group in SELECTED_LABELS:
        sub = plot_df[plot_df["group"].eq(group)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        ax.annotate(group, (row["x"], row["y"]), xytext=(5, -8), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.axvline(0, color="#888888", lw=0.7)
    ax.set_xlabel(f"{channel} DINO {x_label}")
    ax.set_ylabel(f"{channel} DINO {y_label}")
    ax.set_title(f"{channel} DINOv2 target-centroid {method.upper()} map (batch NTC-z)", fontsize=15, weight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    coords = plot_df[["group", "category", "state_class", "x", "y"]].copy()
    coords.insert(0, "method", method)
    coords.insert(0, "channel", channel)
    coords.to_csv(out.with_suffix(".csv"), index=False)


def plot_distance_heatmap(dist: pd.DataFrame, out: Path) -> None:
    channels = list(pd.unique(dist["channel"]))
    fig, axes = plt.subplots(1, len(channels), figsize=(5.4 * len(channels), 4.8), squeeze=False)
    for ax, channel in zip(axes[0], channels):
        sub = dist[dist["channel"].eq(channel)]
        mat = sub.pivot(index="state_a", columns="state_b", values="rms_distance").reindex(index=STATE_ORDER, columns=STATE_ORDER)
        im = ax.imshow(mat.to_numpy(float), cmap="viridis")
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels([STATE_LABELS[s] for s in STATE_ORDER], rotation=45, ha="right")
        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_yticklabels([STATE_LABELS[s] for s in STATE_ORDER])
        ax.set_title(f"{channel} state centroid RMS distance")
        for i in range(len(STATE_ORDER)):
            for j in range(len(STATE_ORDER)):
                val = mat.iloc[i, j]
                ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > np.nanmax(mat.to_numpy()) * 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    import anndata as ad

    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.mm)
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["plate"] = obs["plate"].astype(str)
    table = pd.read_csv(args.table)

    all_dist = []
    all_nn = []
    all_sep = []
    all_targets = []
    for channel, obsm_key in [("C1 brightfield", "X_dino_c1"), ("C24 mitochondrial", "X_dino_c24")]:
        Xraw = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
        Xz = batch_ntc_z(Xraw, obs, args.clip)
        cent = target_centroids(Xz, obs, table)
        plot_df, X, _ = standardize_features(cent)
        plot_df["channel"] = channel
        all_targets.append(plot_df.drop(columns=[c for c in plot_df.columns if c.startswith("f")]))
        all_dist.append(state_distance_table(plot_df, X, channel))
        all_nn.append(nearest_neighbors(plot_df, X, channel, SELECTED_LABELS))
        all_sep.append(separation_summary(plot_df, X, channel))
        slug = "c1" if obsm_key.endswith("c1") else "c24"
        for method in ["pca", "tsne"] + (["umap"] if umap is not None else []):
            suffix = "state_map" if method == "pca" else f"{method}_state_map"
            plot_state_map(plot_df, X, channel, method, figs / f"dino_{slug}_{suffix}.png", args)

    dist = pd.concat(all_dist, ignore_index=True)
    nn = pd.concat(all_nn, ignore_index=True)
    sep = pd.DataFrame(all_sep)
    target_meta = pd.concat(all_targets, ignore_index=True)
    dist.to_csv(out / "dino_state_centroid_distances.csv", index=False)
    nn.to_csv(out / "dino_nearest_neighbors.csv", index=False)
    sep.to_csv(out / "dino_state_separation_summary.csv", index=False)
    target_meta.to_csv(out / "dino_target_centroid_metadata.csv", index=False)
    plot_distance_heatmap(dist, figs / "dino_state_centroid_distance_heatmap.png")

    print("[ba_dino_space] wrote:")
    for path in [
        out / "dino_state_separation_summary.csv",
        out / "dino_state_centroid_distances.csv",
        out / "dino_nearest_neighbors.csv",
        figs / "dino_c1_state_map.png",
        figs / "dino_c24_state_map.png",
        figs / "dino_c1_tsne_state_map.png",
        figs / "dino_c24_tsne_state_map.png",
        figs / "dino_c1_umap_state_map.png",
        figs / "dino_c24_umap_state_map.png",
        figs / "dino_state_centroid_distance_heatmap.png",
    ]:
        if path.exists():
            print(f"  {path}")
    print(sep.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())