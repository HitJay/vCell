"""Tri-modal alignment for the HepG2 EE DRUG-seq screen.

Aligns three well-level modalities into a single, fully-paired ``AnnData``:

1. **Transcriptome** — DRUG-seq mini-bulk expression (from the main-line-D
   processed AnnData): ``counts`` / ``lognorm`` layers and the HVG embeddings.
2. **Imaging** — C24 (TMRM + MitoTracker) DINOv2 features (384-dim, per well)
   plus the vAssay TabPFN Seahorse predictions (``pred_MB`` / ``pred_AUC``).
3. **TMRM phenotype** — the mechanistic axes derived in main line D
   (``pheno_intensity_z``, ``pheno_area_z``, ``pheno_permito_dpsi_z``,
   ``pheno_mitomass_z``) plus KD tier and toxicity flags.

All three are matched on the well key ``(image_plate, well)``, verified to cover
1440/1440 wells with zero loss. The result is written as one AnnData with the
imaging block in ``.obsm`` and the phenotype/prediction columns in ``.obs`` so
downstream models can consume any combination of modalities.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_IMAGE_BASE = "/NNRCC_Image/processed_data/UHYG/2025"
DINO_PREFIX = "DINO_Feature_"
PHENO_COLS = (
    "pheno_intensity_z",
    "pheno_area_z",
    "pheno_permito_dpsi_z",
    "pheno_mitomass_z",
)
#: vAssay channel-combination tags -> what imaging channels / dyes they encode.
#: (verified from 2_tmrm.py + raw images, 2026-06-11)
CHANNEL_MEANING = {
    "C1": "brightfield (ch1, label-free morphology)",
    "C12": "brightfield + TMRM (ch1+ch2)",
    "C14": "brightfield + MitoTracker (ch1+ch4)",
    "C24": "TMRM + MitoTracker (ch2+ch4, mitochondrial)",
}


@dataclass
class AlignConfig:
    processed_adata: str = "data/processed/adata_drugseq_processed.h5ad"
    image_base: str = DEFAULT_IMAGE_BASE
    image_file_key: str = "tmrm_operetta_data_file_name"
    well_key: str = "well"
    #: imaging channel combinations to align (each becomes its own obsm block).
    #: C1 = brightfield/label-free (best as independent modelling input),
    #: C24 = mitochondrial dyes (best Seahorse correlate).
    channel_tags: tuple[str, ...] = ("C1", "C24")
    out_dir: str = "data/processed"
    out_name: str = "adata_multimodal.h5ad"


@dataclass
class AlignResult:
    adata: Any
    summary: dict[str, Any] = field(default_factory=dict)


def _load_imaging(
    plates: list[str], base: str, tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-well DINOv2 features and Seahorse predictions for one channel tag.

    Returns (features_df, preds_df) both keyed on ['image_plate', 'well'].
    """
    feat_rows, pred_rows = [], []
    for p in plates:
        agg = glob.glob(f"{base}/{p}/csv/*DINO2_features_{tag}_ID_aggre.csv")
        rd = glob.glob(f"{base}/{p}/csv/*vAssay_readout_{tag}.csv")
        if agg:
            d = pd.read_csv(agg[0])
            d = d.rename(columns={"ImgID": "well"})
            d["image_plate"] = p
            feat_rows.append(d)
        if rd:
            r = pd.read_csv(rd[0])[["ImgID", "pred_MB", "pred_AUC"]]
            r = r.rename(columns={"ImgID": "well"})
            r["image_plate"] = p
            pred_rows.append(r)
    feats = pd.concat(feat_rows, ignore_index=True)
    preds = pd.concat(pred_rows, ignore_index=True)
    return feats, preds


def align(cfg: AlignConfig) -> AlignResult:
    """Build the tri-modal AnnData."""
    import anndata as ad

    adata = ad.read_h5ad(cfg.processed_adata)
    obs = adata.obs
    obs["image_plate"] = obs[cfg.image_file_key].astype(str)
    obs["well"] = obs[cfg.well_key].astype(str)

    plates = list(pd.unique(obs["image_plate"]))
    key = ["image_plate", "well"]
    left = obs[key].copy()
    left["_row"] = np.arange(len(left))

    channels_meta = {}
    missing_per_channel = {}
    # the primary channel (first tag) supplies the obs-level vassay_pred_* columns
    for ti, tag in enumerate(cfg.channel_tags):
        feats, preds = _load_imaging(plates, cfg.image_base, tag)
        dino_cols = sorted(
            [c for c in feats.columns if c.startswith(DINO_PREFIX)],
            key=lambda c: int(c.split("_")[-1]),
        )
        fmerge = (left.merge(feats[key + dino_cols], on=key, how="left")
                  .sort_values("_row").reset_index(drop=True))
        dino = fmerge[dino_cols].to_numpy(dtype=np.float32)
        n_missing = int(np.isnan(dino).any(axis=1).sum())

        obsm_key = f"X_dino_{tag.lower()}"
        adata.obsm[obsm_key] = dino
        missing_per_channel[tag] = n_missing
        channels_meta[tag] = {
            "obsm_key": obsm_key,
            "dino_dim": len(dino_cols),
            "meaning": CHANNEL_MEANING.get(tag, "unknown"),
            "n_missing": n_missing,
        }

        # per-channel Seahorse predictions
        pmerge = (left.merge(preds[key + ["pred_MB", "pred_AUC"]], on=key, how="left")
                  .sort_values("_row").reset_index(drop=True))
        adata.obs[f"vassay_pred_MB_{tag.lower()}"] = pmerge["pred_MB"].to_numpy(dtype=np.float32)
        adata.obs[f"vassay_pred_AUC_{tag.lower()}"] = pmerge["pred_AUC"].to_numpy(dtype=np.float32)
        if ti == 0:
            # back-compat: primary channel also exposed under the plain names
            adata.obs["vassay_pred_MB"] = pmerge["pred_MB"].to_numpy(dtype=np.float32)
            adata.obs["vassay_pred_AUC"] = pmerge["pred_AUC"].to_numpy(dtype=np.float32)

    adata.uns["multimodal"] = {
        "channel_tags": list(cfg.channel_tags),
        "channels": channels_meta,
        "primary_channel": cfg.channel_tags[0],
        "pheno_cols": list(PHENO_COLS),
        "n_wells": int(adata.n_obs),
    }

    summary = {
        "n_wells": int(adata.n_obs),
        "transcriptome_genes": int(adata.n_vars),
        "transcriptome_hvg": int(adata.obsm["X_lognorm_hvg"].shape[1])
        if "X_lognorm_hvg" in adata.obsm else None,
        "imaging_channels": {t: channels_meta[t]["meaning"] for t in cfg.channel_tags},
        "imaging_obsm_keys": {t: channels_meta[t]["obsm_key"] for t in cfg.channel_tags},
        "n_imaging_missing": missing_per_channel,
        "pheno_cols": [c for c in PHENO_COLS if c in adata.obs],
        "has_vassay_pred": bool(adata.obs["vassay_pred_AUC"].notna().any()),
    }
    return AlignResult(adata=adata, summary=summary)


def write(result: AlignResult, cfg: AlignConfig) -> str:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / cfg.out_name
    result.adata.write_h5ad(p)
    return str(p)
