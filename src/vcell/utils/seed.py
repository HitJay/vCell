"""Reproducibility helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The seed value.
        deterministic: If True, force cuDNN into deterministic mode (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device string (``auto`` picks CUDA when available)."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)
