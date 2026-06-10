"""Smoke + sanity tests for the main-line-D DRUG-seq data-foundation pipeline.

These build a tiny synthetic dataset (counts AnnData + per-field imaging CSVs)
that reproduces the key structure of the real data — within-batch NTC controls,
positive controls with known phenotype directions, a strong/weak/failed
knockdown gradient, the ATP5B->ATP5F1B alias and a toxic well — and assert that
each stage behaves correctly. No external files are touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from vcell.data.drugseq import (  # noqa: E402
    CONTROL_LABEL,
    PrepConfig,
    aggregate_images,
    export,
    plate_wise_ntc_zscore,
    resolve_gene_symbols,
    run_pipeline,
    well_position_convert,
)

GENES = ["GENE1", "GENE2", "GENE3", "ATP5F1B"] + [f"FILL{i}" for i in range(56)]
PLATE_LOCS = [f"r{r:02d}c{c:02d}" for r in (2, 3) for c in range(2, 14)]  # 24


def _well_for(loc: str) -> str:
    return well_position_convert(loc)


def _build_plate(plate_name: str, img_name: str, rng):
    """Build one synthetic plate; returns (counts[24,G], obs_rows, image_df)."""
    groups = (
        ["NTC"] * 6 + ["BAM15"] * 3 + ["MK8722"] * 3
        + ["GENE1"] * 3 + ["GENE2"] * 3 + ["GENE3"] * 3 + ["ATP5B"] * 3
    )
    cats = {"NTC": "NC", "BAM15": "PC", "MK8722": "PC", "ATP5B": "PC",
            "GENE1": "Target", "GENE2": "Target", "GENE3": "Target"}
    gidx = {g: i for i, g in enumerate(GENES)}
    base = rng.poisson(30, size=(24, len(GENES))).astype(np.float32) + 1
    first_gene3 = groups.index("GENE3")

    counts, obs_rows, img_rows = [], [], []
    for w, (loc, grp) in enumerate(zip(PLATE_LOCS, groups)):
        row = base[w].copy()
        if grp == "GENE1":
            row[gidx["GENE1"]] *= 0.1    # strong
        elif grp == "GENE2":
            row[gidx["GENE2"]] *= 0.6    # weak
        elif grp == "GENE3":
            row[gidx["GENE3"]] *= 0.95   # failed
        elif grp == "ATP5B":
            row[gidx["ATP5F1B"]] *= 0.2  # strong, via alias
        counts.append(row)

        well = _well_for(loc)
        toxic = (grp == "GENE3" and w == first_gene3)
        obs_rows.append({
            "sample": f"{plate_name}_{well}", "batch": plate_name,
            "num_umis": float(row.sum() * 1000),
            "num_features": int((row > 0).sum()) + 4000,
            "mt_percentage": 0.05, "group": grp, "plate": plate_name,
            "well": well, "category": cats[grp],
            "tmrm_operetta_data_file_name": img_name,
        })
        for _f in range(3):  # three fields per well
            img_rows.append({
                "img_name": f"{loc}f{_f:02d}p01-x.tiff",
                "ch1_intensity": 150.0, "ch1_area": 200.0,
                "ch2_intensity": 60.0 if grp == "MK8722" else 18.0,  # MK8722 up
                "ch2_area": 30.0 if grp == "BAM15" else 120.0,       # BAM15 down
                "cell_count": 3 if toxic else int(rng.integers(400, 700)),
                "ch4_intensity": 80.0 if grp == "MK8722" else 40.0,  # MK8722 -> biogenesis
                "ch4_area": 55.0,
            })
    return np.vstack(counts), obs_rows, pd.DataFrame(img_rows)


@pytest.fixture()
def synthetic_dataset(tmp_path):
    rng = np.random.default_rng(0)
    image_base = tmp_path / "img"
    image_base.mkdir()
    counts_blocks, obs_all = [], []
    for plate, img in [("P1", "IMG1"), ("P2", "IMG2")]:
        counts, obs_rows, img_df = _build_plate(plate, img, rng)
        d = image_base / img
        d.mkdir()
        img_df.to_csv(d / f"{img}.csv", index=False)
        counts_blocks.append(counts)
        obs_all.extend(obs_rows)

    X = np.vstack(counts_blocks).astype(np.float32)
    obs = pd.DataFrame(obs_all)
    obs.index = obs["sample"].values
    var = pd.DataFrame({"feature": GENES, "symbol": GENES}, index=GENES)
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata_path = tmp_path / "adata.h5ad"
    adata.write_h5ad(adata_path)

    return PrepConfig(
        adata_path=str(adata_path), image_base=str(image_base),
        out_dir=str(tmp_path / "out"), n_hvg=15,
        qc_min_umis=0, qc_min_genes=0, write_h5ad=True, write_npz=True,
    )


def test_well_position_convert():
    assert well_position_convert("r02c02") == "B02"
    assert well_position_convert("r03c11") == "C11"
    assert well_position_convert("garbage") == "unknown"


def test_plate_wise_ntc_zscore_centers_on_ntc():
    rng = np.random.default_rng(1)
    X = rng.normal(5, 2, size=(40, 3))
    batch = np.array(["a"] * 20 + ["b"] * 20)
    ntc = np.zeros(40, dtype=bool)
    ntc[[0, 1, 2, 3, 20, 21, 22, 23]] = True
    X[batch == "b"] += 50  # huge batch shift must be removed
    Z = plate_wise_ntc_zscore(X, batch, ntc)
    # NTC wells should be ~0 mean within each batch
    assert abs(Z[ntc].mean()) < 0.5
    # batch offset removed: per-batch NTC means both ~0
    assert abs(Z[(batch == "a") & ntc].mean()) < 0.5
    assert abs(Z[(batch == "b") & ntc].mean()) < 0.5


def test_resolve_symbols_alias_and_compounds(synthetic_dataset):
    from vcell.data.drugseq import load_adata

    cfg = synthetic_dataset
    adata = load_adata(cfg)
    smap, lookup = resolve_gene_symbols(adata, cfg)
    smap = smap.set_index("group")
    assert smap.loc["ATP5B", "kind"] == "gene"
    assert smap.loc["ATP5B", "resolved_symbol"] == "ATP5F1B"
    assert smap.loc["BAM15", "kind"] == "compound"
    assert smap.loc["MK8722", "kind"] == "compound"
    assert smap.loc["NTC", "kind"] == "control"
    # gene targets are resolvable
    assert "GENE1" in lookup and "ATP5B" in lookup


def test_image_aggregation_recovers_cellcount(synthetic_dataset):
    from vcell.data.drugseq import load_adata

    cfg = synthetic_dataset
    adata = load_adata(cfg)
    img = aggregate_images(adata, cfg)
    # one row per well per imaging plate
    assert "cell_count_sum" in img and "n_fields" in img
    assert (img["n_fields"] == 3).all()
    # the recomputed phenotype ratios exist
    assert "ch2_ch1_area_area_ratio" in img


def test_end_to_end_pipeline_and_directions(synthetic_dataset):
    cfg = synthetic_dataset
    result = run_pipeline(cfg)
    t = result.targets.set_index("group")

    # knockdown tiers
    kd = result.targets.set_index("group")["kd_tier"].to_dict()
    assert kd["GENE1"] == "strong"
    assert kd["GENE3"] == "failed"
    assert kd["ATP5B"] in ("strong", "weak")  # alias resolved & scored

    # positive-control phenotype directions (within-batch NTC z)
    assert t.loc["BAM15", "pheno_area_z"] < -1.0       # uncoupler -> area down
    assert t.loc["MK8722", "pheno_intensity_z"] > 1.0  # energizer -> intensity up

    # MitoTracker (ch4) mechanistic axes: BAM15 lowers per-mito ΔΨm (ch2/ch4),
    # MK8722 raises mito mass (ch4/ch1) rather than per-mito potential.
    assert t.loc["BAM15", "pheno_permito_dpsi_z"] < -1.0
    assert t.loc["MK8722", "pheno_mitomass_z"] > 1.0

    # toxicity flagged for the engineered low-cell-count wells
    assert int(result.wells["tox_flag"].sum()) >= 1

    # summary sanity
    s = result.summary
    assert s["n_wells"] == 48
    assert s["batch_key"] == "plate"

    # export produces vCell-ready npz with NTC -> id 0
    paths = export(result, cfg)
    data = np.load(paths["npz"], allow_pickle=True)
    assert int(data["control_index"]) == 0
    labels = list(data["pert_labels"])
    assert labels[0] == CONTROL_LABEL
    pert = data["pert"]
    ntc_rows = (result.adata.obs["group"].to_numpy() == CONTROL_LABEL)
    assert (pert[ntc_rows] == 0).all()
