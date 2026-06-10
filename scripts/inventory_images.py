#!/usr/bin/env python
"""Inventory the raw Operetta images backing the DRUG-seq × TMRM dataset.

For every well in the processed dataset this builds a machine-readable manifest
mapping ``(plate, well, field, channel) -> raw image path`` so that a vision
foundation model (DINOv2, etc.) can batch-read the images later. It also writes
a per-plate availability summary and a short markdown location guide.

Image layout (verified 2026-06-10), per plate under
``/NNRCC_Image/processed_data/UHYG/2025/<plate>/``:

* ``projection/<field-tiff>``  — projected 16-bit TIFF (4 ch). Two naming styles
  occur across plates and are auto-detected per plate:
    - ``A_hyphen``   : ``r{RR}-c{CC}-f{FF}-ch{N}-01.tiff``        (1 plate)
    - ``B_operetta`` : ``r{RR}c{CC}f{FF}p01-ch{N}sk1fk1fl1.tiff``  (23 plates)
* ``jpg/r{RR}c{CC}f{FF}p01-ch{N}sk1fk1fl1.jpg``    — 8-bit JPG preview (4 ch)
* ``<plate>.csv`` / ``readout.csv``                — per-field CellProfiler-style read-outs
* ``csv/*DINO2_features*``                         — pre-computed DINOv2 features (some plates)

Channels: ch1 = nucleus/cell, ch2 = TMRM (ΔΨm), ch3 = (unused in read-out),
ch4 = MitoTracker (mitochondrial mass).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_IMAGE_BASE = "/NNRCC_Image/processed_data/UHYG/2025"
CHANNELS = (1, 2, 3, 4)
CHANNEL_DYE = {
    1: "nucleus_cell",
    2: "TMRM_membrane_potential",
    3: "unused_in_readout",
    4: "MitoTracker_mito_mass",
}


def well_to_rc(well: str) -> tuple[int, int]:
    """``B02`` -> (row=2, col=2)."""
    m = re.fullmatch(r"([A-Z])(\d+)", str(well))
    if not m:
        raise ValueError(f"bad well {well!r}")
    return ord(m.group(1)) - ord("A") + 1, int(m.group(2))


def tiff_name_hyphen(r: int, c: int, f: int, ch: int) -> str:
    """``A_hyphen`` style: r02-c02-f01-ch2-01.tiff."""
    return f"r{r:02d}-c{c:02d}-f{f:02d}-ch{ch}-01.tiff"


def tiff_name_operetta(r: int, c: int, f: int, ch: int) -> str:
    """``B_operetta`` style: r02c02f01p01-ch2sk1fk1fl1.tiff."""
    return f"r{r:02d}c{c:02d}f{f:02d}p01-ch{ch}sk1fk1fl1.tiff"


def jpg_name(r: int, c: int, f: int, ch: int) -> str:
    return f"r{r:02d}c{c:02d}f{f:02d}p01-ch{ch}sk1fk1fl1.jpg"


def detect_tiff_style(plate_dir: Path) -> str:
    """Look at one TIFF in ``<plate>/projection`` to pick the naming style."""
    proj = plate_dir / "projection"
    if not proj.is_dir():
        return "none"
    for f in proj.iterdir():
        if f.suffix == ".tiff":
            name = f.name
            if re.match(r"r\d+-c\d+-f\d+-ch\d+-\d+\.tiff", name):
                return "A_hyphen"
            if re.match(r"r\d+c\d+f\d+p\d+-ch\d+sk\dfk\dfl\d+\.tiff", name):
                return "B_operetta"
            return "other"
    return "none"


_TIFF_NAMER = {"A_hyphen": tiff_name_hyphen, "B_operetta": tiff_name_operetta}


def build_manifest(
    adata_path: str,
    image_base: str,
    n_fields: int,
    check_exists: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    adata = ad.read_h5ad(adata_path)
    obs = adata.obs[["group", "category", "plate", "well",
                     "tmrm_operetta_data_file_name"]].copy()
    base = Path(image_base)

    rows = []
    for _, w in obs.iterrows():
        plate_img = str(w["tmrm_operetta_data_file_name"])
        r, c = well_to_rc(str(w["well"]))
        pdir = base / plate_img
        style = detect_tiff_style(pdir)
        namer = _TIFF_NAMER.get(style)
        for f in range(1, n_fields + 1):
            for ch in CHANNELS:
                tiff = pdir / "projection" / namer(r, c, f, ch) if namer else None
                jpg = pdir / "jpg" / jpg_name(r, c, f, ch)
                rows.append({
                    "plate_batch": str(w["plate"]),
                    "image_plate": plate_img,
                    "tiff_style": style,
                    "group": str(w["group"]),
                    "category": str(w["category"]),
                    "well": str(w["well"]),
                    "field": f,
                    "channel": ch,
                    "dye": CHANNEL_DYE[ch],
                    "tiff_path": str(tiff) if tiff else "",
                    "jpg_path": str(jpg),
                    "tiff_exists": tiff.exists() if (check_exists and tiff) else None,
                })
    manifest = pd.DataFrame(rows)

    # per-plate availability summary
    grp = manifest.groupby("image_plate")
    summary = pd.DataFrame({
        "n_wells": grp["well"].nunique(),
        "n_images": grp.size(),
    }).reset_index()
    if check_exists:
        summary["n_tiff_present"] = grp["tiff_exists"].sum().astype(int).values
        summary["complete"] = summary["n_images"] == summary["n_tiff_present"]
    return manifest, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adata", default="data/drug-seq/adata.h5ad")
    ap.add_argument("--image-base", default=DEFAULT_IMAGE_BASE)
    ap.add_argument("--n-fields", type=int, default=9)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--check-exists", action="store_true",
                    help="stat every path (slower; verifies availability).")
    args = ap.parse_args(argv)

    manifest, summary = build_manifest(
        args.adata, args.image_base, args.n_fields, args.check_exists
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "image_manifest.csv"
    spath = out / "image_manifest_plate_summary.csv"
    manifest.to_csv(mpath, index=False)
    summary.to_csv(spath, index=False)

    n_plates = manifest["image_plate"].nunique()
    n_wells = manifest["well"].nunique()
    print(f"[manifest] {len(manifest):,} image rows | "
          f"{n_plates} plates × {n_wells} wells × {args.n_fields} fields × {len(CHANNELS)} ch")
    print(f"[manifest] wrote {mpath}")
    print(f"[manifest] wrote {spath}")
    if args.check_exists:
        miss = int((~manifest["tiff_exists"]).sum())
        print(f"[manifest] missing TIFFs: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
