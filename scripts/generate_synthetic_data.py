#!/usr/bin/env python
"""Generate a synthetic perturbation dataset.

Thin wrapper over ``vcell gen-data`` that also works without installing the
package (it injects ``src`` onto sys.path).

Example:
    python scripts/generate_synthetic_data.py --out data/synthetic.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["gen-data", *sys.argv[1:]])
