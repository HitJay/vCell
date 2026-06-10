"""Main line D — DRUG-seq × TMRM data-foundation pipeline.

This module turns the raw HepG2 EE siRNA-knockdown DRUG-seq dataset (well-level
mini-bulk counts) plus the paired TMRM Operetta imaging read-outs into a single,
clean, batch-corrected analysis substrate that the downstream課題 (B/A/C) all
consume.

It is intentionally split into small, individually-testable *stage* functions
that take an :class:`~anndata.AnnData` (and/or DataFrames) and annotate it in
place, plus a :func:`run_pipeline` orchestrator that handles IO.

Design constraints (all verified on the real data, see docs/research_plan):

* Batch and target are ~98% confounded -> every quantity is standardised
  *within batch relative to that batch's NTC* (never raw across batches).
* Knockdown is often weak -> per-target KD efficiency is scored and tiered.
* ``*_cell_count_ratio`` explodes on cell death -> the main phenotype uses the
  cell-count-free ``ch2_ch1_*`` ratios and toxicity is deconvolved using the
  *absolute* ``cell_count`` recovered from the raw Operetta CSVs.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants (data-derived; see docs/research_plan_EE_drugseq.md)
# --------------------------------------------------------------------------- #
CONTROL_LABEL = "NTC"
#: Positive controls that are *small molecules*, not gene knockdowns. They have
#: no matching gene symbol and KD-QC does not apply to them.
POSITIVE_COMPOUNDS = ("BAM15", "MK8722")
#: ``group`` label -> gene symbol used in ``var['symbol']`` (HGNC renames).
SYMBOL_ALIASES = {"ATP5B": "ATP5F1B"}

#: Default location of the raw, field-level Operetta CSVs.
DEFAULT_IMAGE_BASE = "/NNRCC_Image/processed_data/UHYG/2025"

#: Raw per-field imaging columns we aggregate to the well level.
RAW_IMAGE_COLS = (
    "ch1_intensity",
    "ch1_area",
    "ch2_intensity",
    "ch2_area",
    "cell_count",
    "ch4_intensity",
    "ch4_area",
)
#: Imaging channels (confirmed): ch1 = nucleus/cell, ch2 = TMRM (membrane
#: potential ΔΨm), ch4 = MitoTracker (mitochondrial mass). ch2/ch1 convolves
#: membrane potential with mito mass; the MitoTracker channel lets us resolve
#: the two mechanistically (per-mitochondrion ΔΨm vs biogenesis).
PHENO_INTENSITY_AXIS = "ch2_ch1_intensity_area_ratio"  # TMRM per cell (convolved)
PHENO_AREA_AXIS = "ch2_ch1_area_area_ratio"  # high-ΔΨm area per cell (convolved)
PHENO_PERMITO_AXIS = "ch2_ch4_area_area_ratio"  # per-mito ΔΨm (uncoupling); BAM15 down
PHENO_MITOMASS_AXIS = "ch4_ch1_intensity_area_ratio"  # mito mass / biogenesis; MK8722 up

#: Curated EE / mitochondrial genes force-kept in the HVG set so downstream
#: pathway analysis always has signal even if they are not statistically HVG.
EE_PATHWAY_GENES = (
    # uncoupling / thermogenesis
    "UCP1", "UCP2", "UCP3", "PPARGC1A", "PPARGC1B", "ESRRA", "DIO2",
    # AMPK
    "PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "PRKAG2",
    # mito biogenesis / TFs
    "NRF1", "NFE2L2", "TFAM", "MYC",
    # OXPHOS representatives (complex I-V)
    "NDUFA1", "NDUFB8", "NDUFS1", "SDHA", "SDHB", "UQCRC2", "CYC1",
    "COX5A", "COX4I1", "MT-CO1", "ATP5F1A", "ATP5F1B", "ATP5MC1",
    # FAO / substrate
    "CPT1A", "CPT1B", "CPT2", "ACOX1", "PDK4", "ACADM",
    # ANT / carriers / creatine
    "SLC25A4", "SLC25A5", "CKB", "CKMT1A", "CKMT1B",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PrepConfig:
    """All knobs for the data-foundation pipeline."""

    # inputs
    adata_path: str = "data/drug-seq/adata.h5ad"
    image_base: str = DEFAULT_IMAGE_BASE
    image_file_key: str = "tmrm_operetta_data_file_name"

    # column names in adata.obs
    pert_key: str = "group"
    batch_key: str = "plate"  # within-batch unit (falls back to ``batch_fallback``)
    batch_fallback: str = "batch"
    well_key: str = "well"
    category_key: str = "category"

    # expression QC (within-batch robust z on log-library metrics)
    qc_min_umis: float = 50_000.0
    qc_min_genes: int = 3_000
    qc_max_mt: float = 0.30
    qc_mad_k: float = 5.0  # robust outlier threshold (|z_MAD| > k)

    # normalization
    target_sum: float = 1e4  # CP10K (log1p-friendly, MSE-ready)
    n_hvg: int = 2_000
    zscore_clip: float = 10.0  # clip within-NTC expression z (scanpy-style)

    # knockdown scoring
    kd_strong_drop: float = 0.5  # >=50% drop vs NTC -> strong
    kd_weak_drop: float = 0.2  # 20-50% -> weak ; <20% -> failed

    # toxicity
    tox_frac: float = 0.3  # cell_count < this fraction of same-batch NTC median -> toxic
    tox_z_cellcount: float = -2.0  # informational robust log-cell_count z (not the flag)
    tox_min_cellcount: float = 500.0  # absolute well-total floor (cells summed over fields)

    # outputs
    out_dir: str = "data/processed"
    report_dir: str = "output/2026-06-10"
    write_h5ad: bool = True
    write_npz: bool = True

    # behaviour
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrepConfig":
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            warnings.warn(f"Ignoring unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class PrepResult:
    """Outputs of the pipeline."""

    adata: Any  # AnnData (annotated)
    wells: pd.DataFrame
    targets: pd.DataFrame
    symbol_map: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def well_position_convert(row_col: str) -> str:
    """``r02c02`` -> ``B02`` (Operetta plate coordinate to standard well)."""
    m = re.fullmatch(r"r(\d+)c(\d+)", str(row_col))
    if not m:
        return "unknown"
    row_num, col_num = int(m.group(1)), int(m.group(2))
    return f"{chr(ord('A') + row_num - 1)}{col_num:02d}"


def plate_wise_ntc_zscore(
    X: np.ndarray,
    batch: np.ndarray,
    ntc_mask: np.ndarray,
    *,
    robust: bool = False,
    clip: float | None = None,
    min_ntc: int = 2,
    eps: float = 1e-8,
) -> np.ndarray:
    """Standardise every column *within batch, relative to that batch's NTC*.

    For each batch the location/scale are estimated from the NTC wells of that
    same batch. If a batch has fewer than ``min_ntc`` NTC wells we fall back to
    that batch's own mean/std (still within-batch, just not NTC-anchored).

    With ``robust=True`` the location/scale are the median and 1.4826·MAD, which
    is appropriate for skewed, heavy-tailed read-outs (e.g. imaging cell_count,
    phenotype ratios) where a plain mean/std flags far too many wells.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.full_like(X, np.nan)
    for b in pd.unique(batch):
        rows = batch == b
        ntc_rows = rows & ntc_mask
        ref = X[ntc_rows] if int(ntc_rows.sum()) >= min_ntc else X[rows]
        if robust:
            loc = np.median(ref, axis=0)
            scale = 1.4826 * np.median(np.abs(ref - loc), axis=0)
            fallback = ref.std(axis=0)
            scale = np.where(scale < eps, fallback, scale)
        else:
            loc = ref.mean(axis=0)
            scale = ref.std(axis=0)
        scale = np.where(scale < eps, eps, scale)
        Z[rows] = (X[rows] - loc) / scale
    if clip is not None:
        Z = np.clip(Z, -clip, clip)
    return Z


def _robust_z(values: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Within-batch median/MAD robust z for 1-D QC metrics."""
    v = np.asarray(values, dtype=np.float64)
    z = np.full_like(v, np.nan)
    for b in pd.unique(batch):
        rows = batch == b
        med = np.median(v[rows])
        mad = np.median(np.abs(v[rows] - med))
        scale = 1.4826 * mad if mad > 0 else (v[rows].std() or 1.0)
        z[rows] = (v[rows] - med) / scale
    return z


# --------------------------------------------------------------------------- #
# Stage 0 — load & symbol resolution
# --------------------------------------------------------------------------- #
def load_adata(cfg: PrepConfig):
    """Read the raw counts AnnData and choose the within-batch unit."""
    import anndata as ad

    adata = ad.read_h5ad(cfg.adata_path)
    if cfg.batch_key not in adata.obs:
        warnings.warn(
            f"batch_key {cfg.batch_key!r} missing; using {cfg.batch_fallback!r}."
        )
        cfg.batch_key = cfg.batch_fallback
    adata.layers["counts"] = adata.X.copy()
    return adata


def resolve_gene_symbols(
    adata, cfg: PrepConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map each perturbation ``group`` label to a row in ``var``.

    Returns a tidy symbol-map table and a ``{group: var_index}`` lookup used by
    the knockdown scorer. Compounds and the control are recorded but have no
    gene index.
    """
    symbols = adata.var["symbol"].astype(str).to_numpy()
    sym_to_idx: dict[str, int] = {}
    for i, s in enumerate(symbols):
        sym_to_idx.setdefault(s, i)

    groups = list(adata.obs[cfg.pert_key].astype("category").cat.categories)
    rows = []
    group_to_gene_idx: dict[str, int] = {}
    for g in groups:
        if g == CONTROL_LABEL:
            kind, sym, idx = "control", "", None
        elif g in POSITIVE_COMPOUNDS:
            kind, sym, idx = "compound", "", None
        else:
            sym = SYMBOL_ALIASES.get(g, g)
            idx = sym_to_idx.get(sym)
            kind = "gene" if idx is not None else "unmatched"
            if idx is not None:
                group_to_gene_idx[g] = idx
        rows.append(
            {"group": g, "kind": kind, "resolved_symbol": sym,
             "var_index": idx, "matched": idx is not None}
        )
    symbol_map = pd.DataFrame(rows)
    n_unmatched = int(((symbol_map["kind"] == "unmatched")).sum())
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} knockdown target(s) have no matching gene symbol: "
            f"{symbol_map.loc[symbol_map.kind=='unmatched','group'].tolist()}"
        )
    return symbol_map, group_to_gene_idx


# --------------------------------------------------------------------------- #
# Stage 1 — expression QC
# --------------------------------------------------------------------------- #
def expression_qc(adata, cfg: PrepConfig) -> None:
    """Flag (never drop) low-quality wells using within-batch robust z."""
    obs = adata.obs
    batch = obs[cfg.batch_key].to_numpy()
    z_umis = _robust_z(obs["num_umis"].to_numpy(), batch)
    z_genes = _robust_z(obs["num_features"].to_numpy(), batch)
    z_mt = _robust_z(obs["mt_percentage"].to_numpy(), batch)

    hard = (
        (obs["num_umis"].to_numpy() < cfg.qc_min_umis)
        | (obs["num_features"].to_numpy() < cfg.qc_min_genes)
        | (obs["mt_percentage"].to_numpy() > cfg.qc_max_mt)
    )
    outlier = (
        (z_umis < -cfg.qc_mad_k)
        | (z_genes < -cfg.qc_mad_k)
        | (z_mt > cfg.qc_mad_k)
    )
    obs["qc_z_umis"] = z_umis
    obs["qc_z_genes"] = z_genes
    obs["qc_z_mt"] = z_mt
    obs["qc_fail"] = hard | outlier


# --------------------------------------------------------------------------- #
# Stage 2 — normalization + within-batch NTC z
# --------------------------------------------------------------------------- #
def normalize_expression(adata, cfg: PrepConfig) -> None:
    """CP10K + log1p, HVG selection, and within-batch NTC z on the HVG block."""
    import scanpy as sc

    # log-normalised full matrix (kept as a layer)
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    # HVG on log-normalised data, then force-keep EE pathway genes
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_hvg, flavor="seurat")
    force = adata.var["symbol"].astype(str).isin(set(EE_PATHWAY_GENES))
    adata.var["highly_variable"] = adata.var["highly_variable"] | force
    adata.var["ee_pathway_gene"] = force.to_numpy()

    hvg = adata.var["highly_variable"].to_numpy()
    lognorm = adata.layers["lognorm"]
    hvg_block = lognorm[:, hvg]
    hvg_block = hvg_block.toarray() if hasattr(hvg_block, "toarray") else np.asarray(hvg_block)
    adata.obsm["X_lognorm_hvg"] = np.asarray(hvg_block, dtype=np.float32)

    batch = adata.obs[cfg.batch_key].to_numpy()
    ntc_mask = (adata.obs[cfg.pert_key].to_numpy() == CONTROL_LABEL)
    adata.obsm["X_zscore_hvg"] = plate_wise_ntc_zscore(
        adata.obsm["X_lognorm_hvg"], batch, ntc_mask, clip=cfg.zscore_clip
    ).astype(np.float32)
    # restore X to raw counts so downstream consumers are explicit about layers
    adata.X = adata.layers["counts"].copy()


# --------------------------------------------------------------------------- #
# Stage 3 — knockdown efficiency
# --------------------------------------------------------------------------- #
def score_knockdown(
    adata, cfg: PrepConfig, group_to_gene_idx: dict[str, int]
) -> pd.DataFrame:
    """Per-target KD efficiency: within-batch Δ of the target's own gene."""
    lognorm = adata.layers["lognorm"]
    obs = adata.obs
    batch = obs[cfg.batch_key].to_numpy()
    groups = obs[cfg.pert_key].to_numpy()
    ntc_mask = groups == CONTROL_LABEL

    rows = []
    for g, gi in group_to_gene_idx.items():
        col = lognorm[:, gi]
        col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
        z = plate_wise_ntc_zscore(col[:, None], batch, ntc_mask).ravel()
        kd_mask = groups == g
        if kd_mask.sum() == 0:
            continue
        # fractional remaining expression vs same-batch NTC (per KD well's batch)
        frac_remaining = []
        for b in pd.unique(batch[kd_mask]):
            sel = kd_mask & (batch == b)
            ntc_b = ntc_mask & (batch == b)
            ref = col[ntc_b].mean() if ntc_b.any() else col[ntc_mask].mean()
            if ref > 1e-9:
                frac_remaining.append(np.expm1(col[sel]).mean() / np.expm1(ref))
        frac = float(np.mean(frac_remaining)) if frac_remaining else np.nan
        drop = 1.0 - frac if np.isfinite(frac) else np.nan
        if not np.isfinite(drop):
            tier = "unknown"
        elif drop >= cfg.kd_strong_drop:
            tier = "strong"
        elif drop >= cfg.kd_weak_drop:
            tier = "weak"
        else:
            tier = "failed"
        rows.append(
            {cfg.pert_key: g, "n_kd": int(kd_mask.sum()),
             "kd_logfc_z": float(np.nanmean(z[kd_mask])),
             "kd_frac_drop": drop, "kd_tier": tier}
        )
    kd = pd.DataFrame(rows).sort_values("kd_frac_drop", ascending=False)
    # propagate tier to wells
    tier_map = dict(zip(kd[cfg.pert_key], kd["kd_tier"]))
    adata.obs["kd_tier"] = (
        adata.obs[cfg.pert_key].map(tier_map).astype("object").fillna("na")
    )
    return kd


# --------------------------------------------------------------------------- #
# Stage 4 — imaging aggregation, phenotype z, toxicity
# --------------------------------------------------------------------------- #
def aggregate_images(adata, cfg: PrepConfig) -> pd.DataFrame:
    """Field -> well aggregation of the raw Operetta CSVs (recovers cell_count).

    For every imaging plate referenced by the dataset, read its per-field CSV,
    map fields to wells and aggregate. Intensities/areas are averaged; the
    absolute ``cell_count`` is summed (total cells per well) and also averaged;
    within-well dispersion (#fields, CV of cell_count) is kept for QC.
    """
    plates = list(pd.unique(adata.obs[cfg.image_file_key].astype(str)))
    base = Path(cfg.image_base)
    out = []
    missing = []
    for p in plates:
        csv = base / p / f"{p}.csv"
        if not csv.exists():
            missing.append(p)
            continue
        df = pd.read_csv(csv)
        df["plate_location"] = df["img_name"].astype(str).str.extract(r"(r\d+c\d+)")
        df = df.dropna(subset=["plate_location"])
        df["well"] = df["plate_location"].map(well_position_convert)
        agg = df.groupby("well").agg(
            **{c: (c, "mean") for c in RAW_IMAGE_COLS if c in df.columns},
            cell_count_sum=("cell_count", "sum"),
            cell_count_cv=("cell_count", lambda s: float(s.std() / s.mean()) if s.mean() else np.nan),
            n_fields=("cell_count", "size"),
        )
        # recompute the two main phenotype ratios from well-aggregated raw signal
        agg[PHENO_INTENSITY_AXIS] = agg["ch2_intensity"] / agg["ch1_area"].replace(0, np.nan)
        agg[PHENO_AREA_AXIS] = agg["ch2_area"] / agg["ch1_area"].replace(0, np.nan)
        # MitoTracker (ch4) normalised axes — mechanistically resolve the
        # convolved ch2/ch1 signal: per-mitochondrion ΔΨm (ch2/ch4, uncoupling)
        # and mitochondrial mass / biogenesis (ch4/ch1).
        if "ch4_area" in agg and "ch4_intensity" in agg:
            agg[PHENO_PERMITO_AXIS] = agg["ch2_area"] / agg["ch4_area"].replace(0, np.nan)
            agg[PHENO_MITOMASS_AXIS] = agg["ch4_intensity"] / agg["ch1_area"].replace(0, np.nan)
            agg["ch2_ch4_intensity_intensity_ratio"] = agg["ch2_intensity"] / agg["ch4_intensity"].replace(0, np.nan)
            agg["ch4_ch1_area_area_ratio"] = agg["ch4_area"] / agg["ch1_area"].replace(0, np.nan)
        agg[cfg.image_file_key] = p
        out.append(agg.reset_index())
    if missing:
        warnings.warn(f"{len(missing)} imaging plate CSV(s) missing: {missing[:5]}")
    if not out:
        raise FileNotFoundError(
            f"No imaging CSVs found under {cfg.image_base!r}."
        )
    return pd.concat(out, ignore_index=True)


def image_phenotypes_and_toxicity(
    adata, image_well: pd.DataFrame, cfg: PrepConfig
) -> None:
    """Merge well-level imaging onto obs; compute 2-axis phenotype z + toxicity."""
    obs = adata.obs
    key = cfg.image_file_key
    # integer-row merge keeps obs order without relying on index names
    left = obs[[cfg.batch_key, cfg.pert_key, cfg.well_key, key]].copy()
    left["_row"] = np.arange(len(left))
    merged = (
        left.merge(image_well, left_on=[cfg.well_key, key], right_on=["well", key],
                   how="left")
        .sort_values("_row")
        .reset_index(drop=True)
    )

    batch = merged[cfg.batch_key].to_numpy()
    ntc_mask = (merged[cfg.pert_key].to_numpy() == CONTROL_LABEL)

    # phenotype axes (within-batch NTC robust z). ch2/ch1: higher intensity =
    # MK8722-like, lower area = BAM15-like. MitoTracker-resolved: per-mito ΔΨm
    # (ch2/ch4) down = uncoupling; mito mass (ch4/ch1) up = biogenesis.
    pheno_axes = [
        (PHENO_INTENSITY_AXIS, "pheno_intensity_z"),
        (PHENO_AREA_AXIS, "pheno_area_z"),
        (PHENO_PERMITO_AXIS, "pheno_permito_dpsi_z"),
        (PHENO_MITOMASS_AXIS, "pheno_mitomass_z"),
    ]
    for axis, name in pheno_axes:
        if axis not in merged:
            continue
        vals = merged[axis].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        z = np.full_like(vals, np.nan)
        if finite.any():
            filled = np.where(finite, vals, np.nanmedian(vals[finite]))
            z = plate_wise_ntc_zscore(
                filled[:, None], batch, ntc_mask, robust=True
            ).ravel()
            z[~finite] = np.nan
        obs[name] = z

    # toxicity: fraction of same-batch NTC median cell_count. A well that has
    # lost most of its cells (ratio < tox_frac) is flagged toxic — robust and
    # directly interpretable, and it separates true cell death (e.g. PSMC3,
    # ratio~0.2) from real uncoupling phenotypes (BAM15/MK8722, ratio~1.0).
    cc = merged["cell_count_sum"].to_numpy(dtype=np.float64)
    cc_ratio = np.full(len(cc), np.nan)
    for b in pd.unique(batch):
        rows = batch == b
        ntc_b = rows & ntc_mask
        ref = cc[ntc_b] if ntc_b.any() else cc[rows]
        med = np.nanmedian(ref)
        cc_ratio[rows] = cc[rows] / med if med > 0 else np.nan
    # informational robust z on log1p(cell_count) (kept, not used for the flag)
    cc_z = plate_wise_ntc_zscore(
        np.log1p(np.nan_to_num(cc, nan=np.nanmedian(cc)))[:, None],
        batch, ntc_mask, robust=True,
    ).ravel()
    ch1_area = merged.get("ch1_area")
    obs["img_cell_count"] = cc
    obs["img_cell_count_cv"] = merged["cell_count_cv"].to_numpy()
    obs["tox_cellcount_ratio"] = cc_ratio
    obs["tox_z_cellcount"] = cc_z
    if ch1_area is not None:
        obs["img_ch1_area"] = ch1_area.to_numpy()
    # toxic if: lost most cells vs NTC, below absolute floor, or non-finite pheno
    obs["tox_flag"] = (
        (cc_ratio < cfg.tox_frac)
        | (cc < cfg.tox_min_cellcount)
        | (~np.isfinite(obs["pheno_intensity_z"].to_numpy()))
        | (~np.isfinite(obs["pheno_area_z"].to_numpy()))
    )
    # carry optional MitoTracker-resolved ratios if present
    for opt in (PHENO_PERMITO_AXIS, PHENO_MITOMASS_AXIS,
                "ch2_ch4_intensity_intensity_ratio", "ch4_ch1_area_area_ratio",
                "ch4_intensity", "ch4_area"):
        if opt in merged:
            obs[f"img_{opt}"] = merged[opt].to_numpy()


# --------------------------------------------------------------------------- #
# Stage 5 — annotations + export
# --------------------------------------------------------------------------- #
def build_annotations(
    adata, cfg: PrepConfig, kd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the well-level and target-level annotation tables."""
    obs = adata.obs
    well_cols = [
        cfg.pert_key, cfg.category_key, cfg.batch_key, cfg.well_key,
        "qc_fail", "qc_z_umis", "qc_z_genes", "qc_z_mt",
        "kd_tier", "pheno_intensity_z", "pheno_area_z",
        "pheno_permito_dpsi_z", "pheno_mitomass_z",
        "img_cell_count", "tox_cellcount_ratio", "tox_z_cellcount", "tox_flag",
    ]
    wells = obs[[c for c in well_cols if c in obs]].copy()

    # target-level summary (one row per perturbation)
    g = obs.groupby(cfg.pert_key, observed=True)
    agg = {
        "n_wells": g.size(),
        "category": g[cfg.category_key].first(),
        "n_batches": g[cfg.batch_key].nunique(),
        "pheno_intensity_z": g["pheno_intensity_z"].mean(),
        "pheno_area_z": g["pheno_area_z"].mean(),
        "tox_rate": g["tox_flag"].mean(),
        "qc_fail_rate": g["qc_fail"].mean(),
    }
    for opt in ("pheno_permito_dpsi_z", "pheno_mitomass_z"):
        if opt in obs:
            agg[opt] = g[opt].mean()
    targets = pd.DataFrame(agg).reset_index()
    targets = targets.merge(
        kd[[cfg.pert_key, "kd_frac_drop", "kd_tier", "n_kd"]],
        on=cfg.pert_key, how="left",
    )

    # transparent EE hit candidate flag (handed to main line B). With the
    # MitoTracker channel we resolve the mechanism: uncoupling lowers the
    # per-mito ΔΨm (ch2/ch4), biogenesis raises mito mass (ch4/ch1).
    strong = targets["kd_tier"].isin(["strong", "weak"]) | targets["category"].eq("PC")
    clean = targets["tox_rate"] < 0.5
    uncoupler = targets["pheno_area_z"] <= -2.0
    if "pheno_permito_dpsi_z" in targets:
        uncoupler = uncoupler | (targets["pheno_permito_dpsi_z"] <= -2.0)
    if "pheno_mitomass_z" in targets:
        biogenesis = targets["pheno_mitomass_z"] >= 2.0
    else:
        biogenesis = pd.Series(False, index=targets.index)
    energizer = targets["pheno_intensity_z"] >= 2.0
    targets["ee_hit_candidate"] = strong & clean & (uncoupler | biogenesis | energizer)
    targets["ee_hit_direction"] = np.select(
        [uncoupler.fillna(False), biogenesis.fillna(False), energizer.fillna(False)],
        ["uncoupler_like", "biogenesis_like", "energizer_like"],
        default="none",
    )
    sort_key = "pheno_permito_dpsi_z" if "pheno_permito_dpsi_z" in targets else "pheno_area_z"
    return wells, targets.sort_values(sort_key)


def export(result: PrepResult, cfg: PrepConfig) -> dict[str, str]:
    """Write h5ad / npz / CSV products and a JSON summary."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    adata = result.adata

    result.wells.to_csv(out / "wells_annotation.csv")
    result.targets.to_csv(out / "targets_summary.csv", index=False)
    result.symbol_map.to_csv(out / "gene_symbol_map.csv", index=False)
    paths["wells"] = str(out / "wells_annotation.csv")
    paths["targets"] = str(out / "targets_summary.csv")
    paths["symbol_map"] = str(out / "gene_symbol_map.csv")

    if cfg.write_h5ad:
        p = out / "adata_drugseq_processed.h5ad"
        adata.write_h5ad(p)
        paths["h5ad"] = str(p)

    if cfg.write_npz:
        groups = adata.obs[cfg.pert_key].astype("category")
        cats = list(groups.cat.categories)
        # force NTC -> id 0 so vCell's control_index is correct
        if CONTROL_LABEL in cats:
            cats.remove(CONTROL_LABEL)
            cats = [CONTROL_LABEL] + cats
        mapping = {c: i for i, c in enumerate(cats)}
        pert = groups.map(mapping).to_numpy().astype(np.int64)
        p = out / "drugseq_vcell.npz"
        np.savez_compressed(
            p,
            X=adata.obsm["X_lognorm_hvg"].astype(np.float32),
            pert=pert,
            control_index=np.int64(0),
            num_perturbations=np.int64(len(cats)),
            pert_labels=np.array(cats, dtype=object),
        )
        paths["npz"] = str(p)

    with open(out / "prep_summary.json", "w") as fh:
        json.dump(result.summary, fh, indent=2, default=str)
    paths["summary"] = str(out / "prep_summary.json")
    return paths


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: PrepConfig) -> PrepResult:
    """Run all six stages and return the annotated AnnData + tables."""
    np.random.seed(cfg.seed)

    adata = load_adata(cfg)
    symbol_map, group_to_gene_idx = resolve_gene_symbols(adata, cfg)
    expression_qc(adata, cfg)
    normalize_expression(adata, cfg)
    kd = score_knockdown(adata, cfg, group_to_gene_idx)
    image_well = aggregate_images(adata, cfg)
    image_phenotypes_and_toxicity(adata, image_well, cfg)
    wells, targets = build_annotations(adata, cfg, kd)

    summary = {
        "n_wells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_hvg": int(adata.var["highly_variable"].sum()),
        "n_perturbations": int(adata.obs[cfg.pert_key].nunique()),
        "kd_tier_counts": kd["kd_tier"].value_counts().to_dict(),
        "n_qc_fail": int(adata.obs["qc_fail"].sum()),
        "n_tox_flag": int(adata.obs["tox_flag"].sum()),
        "n_ee_hit_candidates": int(targets["ee_hit_candidate"].sum()),
        "ee_hit_directions": targets.loc[
            targets["ee_hit_candidate"], "ee_hit_direction"
        ].value_counts().to_dict(),
        "batch_key": cfg.batch_key,
    }
    result = PrepResult(adata=adata, wells=wells, targets=targets,
                        symbol_map=symbol_map, summary=summary)
    return result
