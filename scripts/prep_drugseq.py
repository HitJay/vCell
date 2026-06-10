#!/usr/bin/env python
"""CLI for the main-line-D DRUG-seq × TMRM data-foundation pipeline.

Examples
--------
    # full run with the default config
    python scripts/prep_drugseq.py --config configs/drugseq_prep.yaml

    # override individual fields
    python scripts/prep_drugseq.py --config configs/drugseq_prep.yaml \
        --set n_hvg=3000 out_dir=data/processed_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from a source checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.data.drugseq import PrepConfig, export, run_pipeline  # noqa: E402


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a YAML PrepConfig.")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="key=value",
        help="Override config fields.",
    )
    ap.add_argument(
        "--no-write", action="store_true",
        help="Run the pipeline but skip writing h5ad/npz (still writes CSVs).",
    )
    args = ap.parse_args(argv)

    cfg = PrepConfig.from_yaml(args.config)
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if not hasattr(cfg, key):
            ap.error(f"Unknown config field: {key!r}")
        setattr(cfg, key, _coerce(value))
    if args.no_write:
        cfg.write_h5ad = False
        cfg.write_npz = False

    print(f"[prep_drugseq] adata   : {cfg.adata_path}")
    print(f"[prep_drugseq] images  : {cfg.image_base}")
    print(f"[prep_drugseq] batch by: {cfg.batch_key}")
    result = run_pipeline(cfg)

    print("\n=== summary ===")
    for k, v in result.summary.items():
        print(f"  {k}: {v}")

    paths = export(result, cfg)
    print("\n=== written ===")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
