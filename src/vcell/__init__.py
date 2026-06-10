"""vCell — virtual cell perturbation-response modelling.

A compact PyTorch implementation of a latent-additive conditional VAE for
predicting single-cell responses to perturbations.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Keep the top-level import light (config is torch-free). Heavier symbols such
# as :class:`vcell.models.VirtualCell` are imported lazily from their modules.
from vcell.utils.config import Config, DataConfig, ModelConfig, TrainConfig

__all__ = [
    "__version__",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
]
