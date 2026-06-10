#!/usr/bin/env python
"""Train a VirtualCell model.

Thin wrapper over ``vcell train`` that also works without installing the
package (it injects ``src`` onto sys.path).

Example:
    python scripts/train.py --config configs/default.yaml --set train.epochs=30
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcell.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
