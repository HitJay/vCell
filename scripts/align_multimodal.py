#!/usr/bin/env python
"""CLI: build the tri-modal aligned AnnData (transcriptome × C24 imaging × TMRM
phenotype) for the HepG2 EE DRUG-seq screen.

    /data/user/QYJI/miniforge3/envs/scvi/bin/python scripts/align_multimodal.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.data.multimodal import AlignConfig, align, write  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed/adata_drugseq_processed.h5ad")
    ap.add_argument("--image-base", default="/NNRCC_Image/processed_data/UHYG/2025")
    ap.add_argument("--channels", nargs="+", default=["C1", "C24"],
                    help="imaging channel combos to align (C1=brightfield, C24=mito).")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--out-name", default="adata_multimodal.h5ad")
    args = ap.parse_args(argv)

    cfg = AlignConfig(
        processed_adata=args.processed,
        image_base=args.image_base,
        channel_tags=tuple(args.channels),
        out_dir=args.out_dir,
        out_name=args.out_name,
    )
    result = align(cfg)
    print("=== tri-modal alignment summary ===")
    for k, v in result.summary.items():
        print(f"  {k}: {v}")
    path = write(result, cfg)
    print(f"\n[align] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
