"""Smoke test for tri-modal alignment (transcriptome × C24 imaging × TMRM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from vcell.data.multimodal import AlignConfig, align  # noqa: E402


@pytest.fixture()
def mini_multimodal(tmp_path):
    # tiny processed AnnData: 2 plates x 3 wells
    wells = ["B02", "B03", "B04"]
    plates = ["IMGX", "IMGY"]
    rows = []
    for pl in plates:
        for w in wells:
            rows.append({"tmrm_operetta_data_file_name": pl, "well": w,
                         "group": "NTC", "plate": pl,
                         "pheno_intensity_z": 0.1, "pheno_area_z": -0.2,
                         "pheno_permito_dpsi_z": 0.0, "pheno_mitomass_z": 0.3,
                         "kd_tier": "na", "tox_flag": False, "img_cell_count": 500.0})
    obs = pd.DataFrame(rows)
    obs.index = [f"{r.tmrm_operetta_data_file_name}_{r.well}" for r in obs.itertuples()]
    X = np.random.default_rng(0).poisson(5, size=(len(obs), 8)).astype(np.float32)
    adata = anndata.AnnData(X=X, obs=obs)
    adata.layers["counts"] = X.copy()
    adata.layers["lognorm"] = np.log1p(X)
    adata.obsm["X_lognorm_hvg"] = X[:, :4].astype(np.float32)
    pa = tmp_path / "proc.h5ad"
    adata.write_h5ad(pa)

    # per-plate C1 (brightfield) + C24 (mito) imaging csvs (aggre features + readout)
    base = tmp_path / "img"
    for pl in plates:
        d = base / pl / "csv"
        d.mkdir(parents=True)
        for tag, auc in [("C1", [108.0, 102.0, 96.0]), ("C24", [110.0, 100.0, 90.0])]:
            feat = pd.DataFrame({"ImgID": wells})
            for i in range(384):
                feat[f"DINO_Feature_{i}"] = np.linspace(0, 1, len(wells)) + i * 0.001
            feat.to_csv(d / f"{pl}_DINO2_features_{tag}_ID_aggre.csv", index=False)
            rd = feat.copy()
            rd["pred_MB"] = [1.3, 1.4, 1.5]
            rd["pred_AUC"] = auc
            rd.to_csv(d / f"{pl}_vAssay_readout_{tag}.csv", index=False)

    return AlignConfig(processed_adata=str(pa), image_base=str(base),
                       channel_tags=("C1", "C24"), out_dir=str(tmp_path / "out"))


def test_alignment_full_coverage(mini_multimodal):
    res = align(mini_multimodal)
    a = res.adata
    # both imaging blocks present and fully aligned
    assert "X_dino_c1" in a.obsm and "X_dino_c24" in a.obsm
    assert a.obsm["X_dino_c1"].shape == (6, 384)
    assert a.obsm["X_dino_c24"].shape == (6, 384)
    assert res.summary["n_imaging_missing"]["C1"] == 0
    assert res.summary["n_imaging_missing"]["C24"] == 0
    # per-channel predictions on obs
    assert "vassay_pred_AUC_c1" in a.obs and "vassay_pred_AUC_c24" in a.obs
    assert a.obs["vassay_pred_AUC"].notna().all()
    # transcriptome + phenotype retained
    assert "lognorm" in a.layers
    assert "pheno_permito_dpsi_z" in a.obs
    # uns metadata
    assert a.uns["multimodal"]["channels"]["C1"]["dino_dim"] == 384
    assert a.uns["multimodal"]["primary_channel"] == "C1"


def test_alignment_preserves_obs_order(mini_multimodal):
    res = align(mini_multimodal)
    a = res.adata
    # first well B02 of IMGX -> C24 pred_AUC 110 ; ordering preserved
    assert abs(float(a.obs["vassay_pred_AUC_c24"].iloc[0]) - 110.0) < 1e-4
    assert abs(float(a.obs["vassay_pred_AUC_c24"].iloc[2]) - 90.0) < 1e-4
    # C1 (primary) exposed under plain name
    assert abs(float(a.obs["vassay_pred_AUC"].iloc[0]) - 108.0) < 1e-4
