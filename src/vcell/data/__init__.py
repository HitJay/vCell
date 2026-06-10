"""Data subpackage: synthetic generation and perturbation datasets."""
from __future__ import annotations

from vcell.data.dataset import (
    DatasetBundle,
    PerturbationDataset,
    build_datasets,
    load_h5ad,
    load_npz,
    make_dataloaders,
)
from vcell.data.synthetic import generate_and_save, generate_synthetic, save_npz

__all__ = [
    "DatasetBundle",
    "PerturbationDataset",
    "build_datasets",
    "load_h5ad",
    "load_npz",
    "make_dataloaders",
    "generate_and_save",
    "generate_synthetic",
    "save_npz",
]
