#!/usr/bin/env python
"""Compare DINOv2 feature-space neighborhoods with database pathway profiles.

The analysis uses external database gene sets (Hallmark/Reactome by default) as
target-level gene-set activity scores. It is designed for distance and nearest-
neighbor analysis across many targets, so it uses a deterministic mean signature
z-score per gene set rather than per-target permutation GSEA.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


STATE_COLORS = {
    "neutral_or_uncertain": "#8c8c8c",
    "uncoupler_like": "#d95f02",
    "mixed_uncoupling_biogenesis": "#7570b3",
    "biogenesis_like": "#1b9e77",
    "toxic_collapse": "#b2182b",
    "energizer_like": "#66a61e",
}
SELECTED_QUERIES = [
    "BAM15",
    "MK8722",
    "PSMC3",
    "ATP5B",
    "SLC25A4",
    "DDI2",
    "TM6SF2",
    "DGAT2",
    "TAGLN",
    "G6PC",
    "KANSL1",
    "RPL8",
    "RFT1",
    "CHRNE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", default="data/processed/adata_multimodal.h5ad")
    parser.add_argument("--target-table", default="output/2026-06-29/pathway_phenotype_correlation/crossmodal_moa_target_table.csv")
    parser.add_argument("--out", default="output/2026-06-30/dino_pathway_similarity")
    parser.add_argument("--libraries", nargs="+", default=["MSigDB_Hallmark_2020", "Reactome_2022"])
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--max-genes", type=int, default=300)
    parser.add_argument("--top-reactome-terms", type=int, default=200)
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument("--clip", type=float, default=10.0)
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def html_table(df: pd.DataFrame, *, max_rows: int = 25, float_format: str = "{:.3f}") -> str:
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else float_format.format(value))
    return view.to_html(index=False, escape=True, classes="data")


def rel(path: Path, out: Path) -> str:
    return html.escape(path.relative_to(out).as_posix())


def dense_matrix(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def batch_ntc_z(X: np.ndarray, obs: pd.DataFrame, clip: float) -> np.ndarray:
    groups = obs["group"].astype(str).to_numpy()
    plates = obs["plate"].astype(str).to_numpy()
    out = np.empty_like(X, dtype=np.float32)
    for plate in pd.unique(plates):
        mask = plates == plate
        ntc = mask & (groups == "NTC")
        ref_mask = ntc if ntc.sum() >= 2 else mask
        ref = X[ref_mask]
        mu = np.nanmean(ref, axis=0)
        sd = np.nanstd(ref, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        out[mask] = (X[mask] - mu) / sd
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def group_mean_matrix(X: np.ndarray, obs: pd.DataFrame, clean_mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    clean_obs = obs.loc[clean_mask].reset_index(drop=True)
    clean_X = X[clean_mask]
    rows = []
    vectors = []
    for group, idx in clean_obs.groupby("group").indices.items():
        sub = clean_obs.iloc[np.asarray(idx)]
        rows.append(
            {
                "group": str(group),
                "category_obs": str(sub["category"].mode().iloc[0]),
                "n_clean_wells_signature": int(len(idx)),
            }
        )
        vectors.append(clean_X[np.asarray(idx)].mean(axis=0))
    return pd.DataFrame(rows), np.vstack(vectors).astype(np.float32)


def load_signatures_and_dino(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, tuple[pd.DataFrame, np.ndarray]], np.ndarray]:
    adata = ad.read_h5ad(args.adata)
    obs = adata.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["plate"] = obs["plate"].astype(str)
    obs["category"] = obs["category"].astype(str)
    obs["qc_fail"] = as_bool(obs["qc_fail"])
    obs["tox_flag"] = as_bool(obs["tox_flag"])
    table = pd.read_csv(args.target_table)
    meta_cols = [
        "group",
        "category",
        "state_class",
        "kd_tier",
        "tox_rate",
        "consensus_score",
        "phenotype_strength",
        "permito",
        "mitomass",
        "area",
        "intensity",
    ]
    meta = table[[col for col in meta_cols if col in table.columns]].drop_duplicates("group")

    clean_mask = (obs["category"].isin(["Target", "PC"]) & (~obs["qc_fail"]) & (~obs["tox_flag"])).to_numpy()
    X_lognorm = dense_matrix(adata.layers["lognorm"])
    X_expr_z = batch_ntc_z(X_lognorm, obs, args.clip)
    sig_meta, sig_matrix = group_mean_matrix(X_expr_z, obs, clean_mask)
    sig_meta = sig_meta.merge(meta, on="group", how="left")
    sig_meta["category"] = sig_meta["category"].fillna(sig_meta["category_obs"])

    dino = {}
    for channel, obsm_key in [("C1 brightfield", "X_dino_c1"), ("C24 mitochondrial", "X_dino_c24")]:
        X_dino = dense_matrix(adata.obsm[obsm_key])
        X_dino_z = batch_ntc_z(X_dino, obs, clip=8.0)
        dino_meta, dino_matrix = group_mean_matrix(X_dino_z, obs, clean_mask)
        dino_meta = dino_meta.merge(meta, on="group", how="left")
        dino_meta["category"] = dino_meta["category"].fillna(dino_meta["category_obs"])
        keep = dino_meta["category"].isin(["Target", "PC"])
        scaled = StandardScaler().fit_transform(dino_matrix[keep.to_numpy()])
        dino[channel] = (dino_meta.loc[keep].reset_index(drop=True), scaled.astype(np.float32))
    symbols = adata.var["symbol"].astype(str).to_numpy()
    return sig_meta, sig_matrix, symbols, dino, clean_mask


def fetch_gene_sets(libraries: list[str], symbols: np.ndarray, args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    symbol_set = set(symbols)
    libraries_out: dict[str, dict[str, list[str]]] = {}
    for library in libraries:
        raw = gp.get_library(name=library, organism="Human")
        filtered = {}
        for term, genes in raw.items():
            present = sorted({gene for gene in genes if gene in symbol_set})
            if args.min_genes <= len(present) <= args.max_genes:
                filtered[term] = present
        libraries_out[library] = filtered
    return libraries_out


def score_gene_sets(
    sig_meta: pd.DataFrame,
    sig_matrix: np.ndarray,
    symbols: np.ndarray,
    gene_sets: dict[str, dict[str, list[str]]],
    out: Path,
) -> dict[str, dict[str, object]]:
    symbol_to_idx: dict[str, int] = {}
    for i, symbol in enumerate(symbols):
        symbol_to_idx.setdefault(symbol, i)
    scored = {}
    for library, terms in gene_sets.items():
        labels = []
        sizes = []
        columns = []
        for term, genes in terms.items():
            idx = [symbol_to_idx[gene] for gene in genes if gene in symbol_to_idx]
            if not idx:
                continue
            values = sig_matrix[:, idx]
            mean_z = values.mean(axis=1)
            # A lightweight activity statistic for ranking terms. We keep the
            # raw mean_z for distances and this z/p only for top-term summaries.
            stat = mean_z * np.sqrt(len(idx))
            labels.append(term)
            sizes.append(len(idx))
            columns.append(mean_z.astype(np.float32))
            if len(labels) == 0:
                continue
        score = np.vstack(columns).T if columns else np.empty((len(sig_meta), 0), dtype=np.float32)
        term_meta = pd.DataFrame({"library": library, "term": labels, "n_genes_in_data": sizes})
        score_df = pd.concat([sig_meta[["group", "category", "state_class"]].reset_index(drop=True), pd.DataFrame(score, columns=labels)], axis=1)
        long = score_df.melt(id_vars=["group", "category", "state_class"], var_name="term", value_name="mean_signature_z")
        long = long.merge(term_meta, on="term", how="left")
        long["approx_z"] = long["mean_signature_z"] * np.sqrt(long["n_genes_in_data"].clip(lower=1))
        long["approx_p_two_sided"] = 2 * norm.sf(long["approx_z"].abs())
        long.to_csv(out / f"database_gene_set_scores_{safe_name(library)}.csv", index=False)
        term_meta.to_csv(out / f"database_gene_set_terms_{safe_name(library)}.csv", index=False)
        scored[library] = {"score": score, "terms": labels, "term_meta": term_meta, "long": long}
    return scored


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def select_analysis_terms(scored: dict[str, dict[str, object]], sig_meta: pd.DataFrame, args: argparse.Namespace) -> dict[str, tuple[list[str], np.ndarray]]:
    target_mask = sig_meta["category"].eq("Target").to_numpy()
    selected = {}
    for library, payload in scored.items():
        score = payload["score"]
        terms = list(payload["terms"])
        if score.shape[1] == 0:
            continue
        if "reactome" in library.lower() and score.shape[1] > args.top_reactome_terms:
            variances = np.nanvar(score[target_mask], axis=0)
            keep_idx = np.argsort(variances)[::-1][: args.top_reactome_terms]
        else:
            keep_idx = np.arange(score.shape[1])
        selected_terms = [terms[i] for i in keep_idx]
        selected_score = score[:, keep_idx]
        selected[library] = (selected_terms, selected_score)
    if len(selected) >= 2:
        combined_terms = []
        combined_scores = []
        for library, (terms, score) in selected.items():
            combined_terms.extend([f"{library}: {term}" for term in terms])
            combined_scores.append(score)
        selected["Combined_database_selected"] = (combined_terms, np.hstack(combined_scores))
    return selected


def align_target_matrices(
    sig_meta: pd.DataFrame,
    pathway_score: np.ndarray,
    dino_meta: pd.DataFrame,
    dino_matrix: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sig_target = sig_meta[sig_meta["category"].eq("Target")].reset_index(drop=True)
    dino_target = dino_meta[dino_meta["category"].eq("Target")].reset_index(drop=True)
    groups = sorted(set(sig_target["group"]) & set(dino_target["group"]))
    sig_idx = sig_target.set_index("group").loc[groups].index
    dino_idx = dino_target.set_index("group").loc[groups].index
    sig_positions = sig_target.reset_index().set_index("group").loc[sig_idx, "index"].to_numpy()
    dino_positions = dino_target.reset_index().set_index("group").loc[dino_idx, "index"].to_numpy()
    meta = sig_target.iloc[sig_positions].reset_index(drop=True)
    return meta, pathway_score[sig_target.index.to_numpy()[sig_positions]], dino_matrix[dino_positions]


def upper_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def zscore_columns(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full_like(pvalues, np.nan)
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


def analyze_distance_and_neighbors(
    sig_meta: pd.DataFrame,
    selected_scores: dict[str, tuple[list[str], np.ndarray]],
    dino: dict[str, tuple[pd.DataFrame, np.ndarray]],
    args: argparse.Namespace,
    out: Path,
) -> dict[str, pd.DataFrame]:
    distance_rows = []
    predict_rows = []
    similarity_rows = []
    overlap_rows = []
    example_rows = []
    pair_sample_rows = []
    for library, (terms, score) in selected_scores.items():
        for channel, (dino_meta, dino_matrix) in dino.items():
            meta, pathway_target, dino_target = align_target_matrices(sig_meta, score, dino_meta, dino_matrix)
            pathway_z = zscore_columns(pathway_target)
            dino_dist = pairwise_distances(dino_target, metric="euclidean")
            pathway_dist = pairwise_distances(pathway_z, metric="euclidean")
            dino_values = upper_values(dino_dist)
            pathway_values = upper_values(pathway_dist)
            rho, pvalue = spearmanr(dino_values, pathway_values)
            distance_rows.append(
                {
                    "library": library,
                    "channel": channel,
                    "n_targets": len(meta),
                    "n_terms": len(terms),
                    "n_pairs": len(dino_values),
                    "spearman_distance_rho": rho,
                    "spearman_distance_p": pvalue,
                }
            )
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(dino_values), size=min(5000, len(dino_values)), replace=False)
            for i in sample_idx:
                pair_sample_rows.append(
                    {
                        "library": library,
                        "channel": channel,
                        "dino_distance": float(dino_values[i]),
                        "pathway_distance": float(pathway_values[i]),
                    }
                )

            dino_order = np.argsort(dino_dist, axis=1)
            path_order = np.argsort(pathway_dist, axis=1)
            k = min(args.neighbor_k, len(meta) - 1)
            neighbor_idx = np.asarray([row[row != i][:k] for i, row in enumerate(dino_order)])
            pred = np.vstack([pathway_target[idx].mean(axis=0) for idx in neighbor_idx])
            term_rhos = []
            term_ps = []
            for term_idx, term in enumerate(terms):
                actual = pathway_target[:, term_idx]
                predicted = pred[:, term_idx]
                if np.nanstd(actual) < 1e-10 or np.nanstd(predicted) < 1e-10:
                    term_rho, term_p = np.nan, np.nan
                else:
                    term_rho, term_p = spearmanr(actual, predicted)
                term_rhos.append(term_rho)
                term_ps.append(term_p)
                predict_rows.append(
                    {
                        "library": library,
                        "channel": channel,
                        "term": term,
                        "n_targets": len(meta),
                        "neighbor_k": k,
                        "spearman_actual_vs_dino_neighbor_mean": term_rho,
                        "p_value": term_p,
                    }
                )

            path_sim = 1 - pairwise_distances(pathway_z, metric="cosine")
            np.fill_diagonal(path_sim, np.nan)
            for i, row in meta.iterrows():
                dino_neighbors = neighbor_idx[i]
                path_neighbors = path_order[i][path_order[i] != i][:k]
                overlap = len(set(dino_neighbors).intersection(set(path_neighbors))) / k
                neighbor_sim = float(np.nanmean(path_sim[i, dino_neighbors]))
                random_sim = float(np.nanmean(path_sim[i]))
                similarity_rows.append(
                    {
                        "library": library,
                        "channel": channel,
                        "group": row["group"],
                        "state_class": row.get("state_class", np.nan),
                        "neighbor_k": k,
                        "mean_pathway_cosine_of_dino_neighbors": neighbor_sim,
                        "mean_pathway_cosine_all_nonself": random_sim,
                        "pathway_similarity_lift": neighbor_sim - random_sim,
                    }
                )
                overlap_rows.append(
                    {
                        "library": library,
                        "channel": channel,
                        "group": row["group"],
                        "state_class": row.get("state_class", np.nan),
                        "neighbor_k": k,
                        "dino_pathway_topk_neighbor_overlap_fraction": overlap,
                        "random_expected_overlap_fraction": k / (len(meta) - 1),
                    }
                )

            query_pool = [query for query in SELECTED_QUERIES if query in set(meta["group"])]
            strong = meta.sort_values("phenotype_strength", ascending=False).head(8)["group"].astype(str).tolist()
            for query in sorted(set(query_pool + strong)):
                i = int(meta.index[meta["group"].eq(query)][0])
                top_terms_query = top_terms(pathway_target[i], terms)
                for rank, j in enumerate(neighbor_idx[i][:5], start=1):
                    example_rows.append(
                        {
                            "library": library,
                            "channel": channel,
                            "query": query,
                            "query_state": meta.iloc[i].get("state_class", np.nan),
                            "rank": rank,
                            "neighbor": meta.iloc[j]["group"],
                            "neighbor_state": meta.iloc[j].get("state_class", np.nan),
                            "dino_distance": float(dino_dist[i, j]),
                            "pathway_cosine": float(path_sim[i, j]),
                            "query_top_terms": "; ".join(top_terms_query),
                            "neighbor_top_terms": "; ".join(top_terms(pathway_target[j], terms)),
                        }
                    )
    predict = pd.DataFrame(predict_rows)
    if not predict.empty:
        predict["fdr_bh"] = predict.groupby(["library", "channel"])["p_value"].transform(lambda x: fdr_bh(x.to_numpy()))
    outputs = {
        "distance": pd.DataFrame(distance_rows),
        "predictability": predict,
        "similarity": pd.DataFrame(similarity_rows),
        "overlap": pd.DataFrame(overlap_rows),
        "examples": pd.DataFrame(example_rows),
        "pair_samples": pd.DataFrame(pair_sample_rows),
    }
    for name, df in outputs.items():
        df.to_csv(out / f"dino_pathway_{name}.csv", index=False)
    return outputs


def top_terms(values: np.ndarray, terms: list[str], n: int = 3) -> list[str]:
    if len(terms) == 0:
        return []
    order = np.argsort(values)[::-1][:n]
    return [f"{short_term(terms[i])} ({values[i]:.2f})" for i in order]


def short_term(term: str, max_len: int = 58) -> str:
    term = term.replace("HALLMARK_", "").replace(" Reactome", "")
    return term if len(term) <= max_len else term[: max_len - 1] + "..."


def plot_distance_bars(distance: pd.DataFrame, figs: Path) -> Path:
    data = distance.copy()
    data["label"] = data["library"].map(short_term) + "\n" + data["channel"]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    colors = ["#4c78a8" if "C1" in channel else "#f58518" for channel in data["channel"]]
    ax.bar(np.arange(len(data)), data["spearman_distance_rho"], color=colors, alpha=0.82)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho: DINO distance vs pathway distance")
    ax.set_title("Do DINO-near targets have similar database pathway profiles?")
    fig.tight_layout()
    path = figs / "dino_pathway_distance_correlation_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_distance_hexbin(samples: pd.DataFrame, figs: Path) -> Path:
    panels = samples.groupby(["library", "channel"]).ngroup().nunique()
    ncols = 2
    nrows = int(np.ceil(panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, ((library, channel), sub) in zip(axes.ravel(), samples.groupby(["library", "channel"])):
        ax.axis("on")
        hb = ax.hexbin(sub["dino_distance"], sub["pathway_distance"], gridsize=36, cmap="YlGnBu", mincnt=1)
        ax.set_title(f"{short_term(library)} / {channel}", fontsize=10)
        ax.set_xlabel("DINO distance")
        ax.set_ylabel("Pathway distance")
        fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    path = figs / "dino_vs_pathway_distance_hexbin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_predictability(predict: pd.DataFrame, figs: Path) -> Path:
    data = predict.copy()
    data = data[np.isfinite(data["spearman_actual_vs_dino_neighbor_mean"])]
    top = data.groupby("term")["spearman_actual_vs_dino_neighbor_mean"].max().sort_values(ascending=False).head(24).index
    plot = data[data["term"].isin(top)].copy()
    plot["row"] = plot["term"].map(short_term)
    plot["col"] = plot["library"].map(short_term) + " / " + plot["channel"]
    mat = plot.pivot_table(index="row", columns="col", values="spearman_actual_vs_dino_neighbor_mean", aggfunc="max")
    mat = mat.reindex(index=[short_term(t) for t in top])
    fig, ax = plt.subplots(figsize=(12.2, 8.2))
    image = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Pathway activities predictable from DINO nearest neighbors")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = figs / "dino_neighbor_pathway_predictability_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_similarity_lift(similarity: pd.DataFrame, figs: Path) -> Path:
    data = similarity.copy()
    groups = list(data.groupby(["library", "channel"]).groups)
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    values = [data[(data["library"].eq(lib)) & (data["channel"].eq(ch))]["pathway_similarity_lift"].dropna().to_numpy() for lib, ch in groups]
    ax.boxplot(values, tick_labels=[f"{short_term(lib)}\n{ch}" for lib, ch in groups], showfliers=False)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylabel("DINO-neighbor pathway cosine lift vs all nonself")
    ax.set_title("Does DINO nearest-neighbor retrieval enrich pathway-similar targets?")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = figs / "dino_neighbor_pathway_similarity_lift.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def render_html(out: Path, outputs: dict[str, pd.DataFrame], figures: dict[str, Path]) -> Path:
    distance = outputs["distance"].copy().sort_values("spearman_distance_rho", ascending=False)
    predict = outputs["predictability"].copy().sort_values("spearman_actual_vs_dino_neighbor_mean", ascending=False)
    similarity_summary = outputs["similarity"].groupby(["library", "channel"]).agg(
        median_pathway_similarity_lift=("pathway_similarity_lift", "median"),
        mean_pathway_similarity_lift=("pathway_similarity_lift", "mean"),
    ).reset_index()
    overlap_summary = outputs["overlap"].groupby(["library", "channel"]).agg(
        mean_topk_overlap=("dino_pathway_topk_neighbor_overlap_fraction", "mean"),
        random_expected_overlap=("random_expected_overlap_fraction", "mean"),
    ).reset_index()
    best_row = distance.iloc[0]
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DINOv2 Feature Distance vs Database Pathway Similarity</title>
  <style>
    :root {{ --ink:#17202a; --muted:#64748b; --line:#d8dee8; --bg:#f5f7f2; --panel:#fff; --accent:#0f766e; }}
    body {{ margin:0; font-family:Avenir Next, Noto Sans, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:36px 48px 28px; background:linear-gradient(135deg,#f8fafc 0%,#e8efe7 58%,#f7ead8 100%); border-bottom:1px solid var(--line); }}
    main {{ max-width:1180px; margin:0 auto; padding:26px 24px 54px; }}
    h1 {{ margin:0 0 10px; font-size:32px; letter-spacing:0; }} h2 {{ margin:0 0 14px; font-size:22px; }}
    p {{ line-height:1.62; }} .lead {{ max-width:960px; color:#334155; }}
    * {{ box-sizing:border-box; }}
    section {{ min-width:0; background:var(--panel); border:1px solid var(--line); margin:18px 0; padding:22px; box-shadow:0 10px 28px rgba(31,41,55,.05); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:18px; }}
    .metric {{ background:rgba(255,255,255,.72); border:1px solid var(--line); padding:14px 16px; }} .metric b {{ display:block; font-size:24px; }} .metric span {{ color:var(--muted); font-size:13px; }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; align-items:start; }}
    .two-col > *, .grid > * {{ min-width:0; }}
    figure {{ margin:0; }} figure img {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }} figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
    .callout {{ border-left:4px solid var(--accent); background:#ecfdf5; padding:13px 15px; color:#134e4a; }}
    .table-scroll {{ max-width:100%; min-width:0; overflow-x:auto; border:1px solid var(--line); }} table.data {{ border-collapse:collapse; width:max-content; min-width:100%; max-width:none; font-size:13px; }} table.data th, table.data td {{ border-bottom:1px solid #e5e7eb; padding:7px 9px; text-align:right; white-space:nowrap; vertical-align:top; }} table.data th:first-child, table.data td:first-child {{ text-align:left; }} table.data td:nth-child(3), table.data td:last-child {{ max-width:420px; white-space:normal; overflow-wrap:anywhere; }} table.data th {{ background:#f8fafc; color:#334155; }}
    code {{ background:#f1f5f9; padding:2px 5px; }}
    @media (max-width:860px) {{ header {{ padding:26px 22px; }} main {{ padding:18px 14px 42px; }} .grid,.two-col {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>DINOv2 Feature Distance vs Database Pathway Similarity</h1>
  <p class="lead">用 Hallmark/Reactome 外部数据库 gene set 给每个 perturbation 建 pathway activity profile，再检验 DINOv2 C1/C24 空间距离是否能找回 pathway-similar perturbations。</p>
  <div class="grid">
    <div class="metric"><b>{int(distance['n_targets'].max())}</b><span>target perturbations</span></div>
    <div class="metric"><b>{int(distance['n_terms'].max())}</b><span>max database terms in a profile</span></div>
    <div class="metric"><b>{best_row['spearman_distance_rho']:.3f}</b><span>best DINO-pathway distance rho</span></div>
  </div>
</header>
<main>
  <section>
    <h2>核心结论</h2>
    <div class="callout">DINO 空间和数据库 pathway profile 有可检测但偏弱的对应关系。正的 distance rho 表示 DINO 距离越近，pathway profile 越相似；当前最佳组合是 {html.escape(str(best_row['library']))} / {html.escape(str(best_row['channel']))}: rho={best_row['spearman_distance_rho']:.3f}。</div>
    <p>这个结果更像“DINO 可以提供 pathway-neighborhood hint”，而不是“DINO 空间能直接重建 pathway annotation”。因此更适合用于找候选相似 perturbation、生成机制假设，再用 DRUG-seq enrichment 或 wet-lab readout 审核。</p>
  </section>

  <section class="two-col">
    <figure><img src="{rel(figures['bars'], out)}" alt="distance correlation"><figcaption>DINO pairwise distance vs database pathway pairwise distance。</figcaption></figure>
    <figure><img src="{rel(figures['lift'], out)}" alt="pathway similarity lift"><figcaption>DINO nearest neighbors 的 pathway cosine similarity 相对全体 nonself baseline 的提升。</figcaption></figure>
  </section>

  <section>
    <h2>Distance Scatter</h2>
    <figure><img src="{rel(figures['hexbin'], out)}" alt="DINO vs pathway distance"><figcaption>每个点是一个 target pair；图中为抽样后的 pairwise 距离密度。</figcaption></figure>
  </section>

  <section>
    <h2>DINO-neighbor 可预测的 pathway terms</h2>
    <figure><img src="{rel(figures['predictability'], out)}" alt="predictability heatmap"><figcaption>每个 term 的实际 pathway activity vs DINO 近邻均值的 target-level Spearman。</figcaption></figure>
  </section>

  <section class="two-col">
    <div><h2>Distance Correlations</h2><div class="table-scroll">{html_table(distance, max_rows=20)}</div></div>
    <div><h2>Neighbor Similarity Lift</h2><div class="table-scroll">{html_table(similarity_summary, max_rows=20)}</div></div>
  </section>

  <section class="two-col">
    <div><h2>DINO vs Pathway Neighbor Overlap</h2><div class="table-scroll">{html_table(overlap_summary, max_rows=20)}</div></div>
    <div><h2>Top Predictable Terms</h2><div class="table-scroll">{html_table(predict[['library','channel','term','spearman_actual_vs_dino_neighbor_mean','fdr_bh']], max_rows=18)}</div></div>
  </section>

  <section>
    <h2>Example DINO Neighbors With Pathway Profiles</h2>
    <div class="table-scroll">{html_table(outputs['examples'], max_rows=40)}</div>
  </section>

  <section>
    <h2>Files</h2>
    <p>Core outputs: <code>dino_pathway_distance.csv</code>, <code>dino_pathway_predictability.csv</code>, <code>dino_pathway_similarity.csv</code>, <code>dino_pathway_overlap.csv</code>, <code>dino_pathway_examples.csv</code>.</p>
    <p>Database score matrices are stored as <code>database_gene_set_scores_*.csv</code>. Scores are external database gene-set activity from plate NTC-z target signatures, not permutation GSEA NES.</p>
  </section>
</main>
</body>
</html>
"""
    path = out / "dino_pathway_similarity_report.html"
    path.write_text(text, encoding="utf-8")
    return path


def make_figures(outputs: dict[str, pd.DataFrame], out: Path) -> dict[str, Path]:
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return {
        "bars": plot_distance_bars(outputs["distance"], figs),
        "hexbin": plot_distance_hexbin(outputs["pair_samples"], figs),
        "predictability": plot_predictability(outputs["predictability"], figs),
        "lift": plot_similarity_lift(outputs["similarity"], figs),
    }


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    sig_meta, sig_matrix, symbols, dino, _ = load_signatures_and_dino(args)
    gene_sets = fetch_gene_sets(args.libraries, symbols, args)
    scored = score_gene_sets(sig_meta, sig_matrix, symbols, gene_sets, out)
    selected = select_analysis_terms(scored, sig_meta, args)
    outputs = analyze_distance_and_neighbors(sig_meta, selected, dino, args, out)
    figures = make_figures(outputs, out)
    report = render_html(out, outputs, figures)
    print(f"[dino_pathway_similarity] wrote {report}")
    print(outputs["distance"].sort_values("spearman_distance_rho", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())