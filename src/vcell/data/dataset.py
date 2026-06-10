"""Dataset, loaders and dataset construction for vCell.

A :class:`PerturbationDataset` holds an expression matrix together with the
perturbation id and dose for every cell. :func:`build_datasets` turns a
:class:`~vcell.utils.config.DataConfig` into train/val splits, transparently
falling back to synthetic data when no ``data_path`` is given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vcell.data.synthetic import generate_synthetic
from vcell.utils.config import DataConfig


class PerturbationDataset(Dataset):
    """In-memory dataset of perturbed single-cell profiles."""

    def __init__(
        self,
        X: np.ndarray,
        pert: np.ndarray,
        dose: np.ndarray | None = None,
        control_index: int = 0,
    ) -> None:
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.pert = torch.as_tensor(np.asarray(pert), dtype=torch.long)
        if dose is None:
            dose = np.ones(len(self.pert), dtype=np.float32)
        self.dose = torch.as_tensor(np.asarray(dose), dtype=torch.float32)
        self.control_index = int(control_index)
        if not (len(self.X) == len(self.pert) == len(self.dose)):
            raise ValueError("X, pert and dose must have matching lengths.")
        if self.X.ndim != 2:
            raise ValueError("X must be a 2-D (cells x genes) array.")

    @property
    def n_genes(self) -> int:
        return int(self.X.shape[1])

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "expression": self.X[i],
            "pert": self.pert[i],
            "dose": self.dose[i],
        }

    # -- evaluation helpers ---------------------------------------------------
    def control_matrix(self) -> torch.Tensor:
        """All control cells (perturbation == control_index)."""
        return self.X[self.pert == self.control_index]

    def perturbation_mean(self, k: int) -> torch.Tensor | None:
        """Mean expression of cells with perturbation ``k`` (or None if empty)."""
        mask = self.pert == k
        if int(mask.sum()) == 0:
            return None
        return self.X[mask].mean(dim=0)

    def present_perturbations(self) -> list[int]:
        """Sorted unique perturbation ids present in this split."""
        return sorted(int(p) for p in torch.unique(self.pert).tolist())


@dataclass
class DatasetBundle:
    """Train/val datasets plus shape metadata and optional ground truth."""

    train: PerturbationDataset
    val: PerturbationDataset
    n_genes: int
    num_perturbations: int
    control_index: int
    meta: dict[str, Any] = field(default_factory=dict)


def _split_indices(
    n: int, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    return perm[n_val:], perm[:n_val]


def load_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int, dict[str, Any]]:
    """Load a dataset previously written by :func:`synthetic.save_npz`."""
    data = np.load(path, allow_pickle=False)
    X = data["X"]
    pert = data["pert"]
    dose = data["dose"] if "dose" in data.files else None
    control_index = int(data["control_index"]) if "control_index" in data.files else 0
    num_perturbations = (
        int(data["num_perturbations"])
        if "num_perturbations" in data.files
        else int(pert.max()) + 1
    )
    meta: dict[str, Any] = {}
    for key in ("effects", "base_mean"):
        if key in data.files:
            meta[key] = data[key]
    return X, pert, dose, control_index, num_perturbations, meta


def load_h5ad(
    path: str | Path, pert_key: str, control_label: str = "control"
) -> tuple[np.ndarray, np.ndarray, None, int, int, dict[str, Any]]:
    """Load an AnnData ``.h5ad`` file (requires the optional ``scrna`` extra)."""
    try:
        import anndata
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading .h5ad requires anndata: pip install -e '.[scrna]'"
        ) from exc

    adata = anndata.read_h5ad(path)
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, dtype=np.float32)

    if pert_key not in adata.obs:
        raise KeyError(f"pert_key {pert_key!r} not found in adata.obs.")
    labels = adata.obs[pert_key].astype("category")
    categories = list(labels.cat.categories)
    if control_label in categories:  # force control to id 0
        categories.remove(control_label)
        categories = [control_label] + categories
    mapping = {cat: i for i, cat in enumerate(categories)}
    pert = labels.map(mapping).to_numpy().astype(np.int64)
    return X, pert, None, 0, len(categories), {}


def build_datasets(cfg: DataConfig) -> DatasetBundle:
    """Build train/val :class:`PerturbationDataset` splits from a config."""
    if cfg.data_path:
        path = Path(cfg.data_path)
        if path.suffix == ".npz":
            X, pert, dose, control_index, num_perturbations, meta = load_npz(path)
        elif path.suffix == ".h5ad":
            X, pert, dose, control_index, num_perturbations, meta = load_h5ad(
                path, cfg.pert_key
            )
        else:
            raise ValueError(f"Unsupported data file type: {path.suffix!r}")
    else:
        data = generate_synthetic(
            n_genes=cfg.n_genes,
            num_perturbations=cfg.num_perturbations,
            n_cells_per_pert=cfg.n_cells_per_pert,
            effect_sparsity=cfg.effect_sparsity,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
        )
        X, pert, dose = data["X"], data["pert"], data["dose"]
        control_index = int(data["control_index"])
        num_perturbations = int(data["num_perturbations"])
        meta = {"effects": data["effects"], "base_mean": data["base_mean"]}

    train_idx, val_idx = _split_indices(len(X), cfg.val_fraction, cfg.seed)
    dose_tr = None if dose is None else dose[train_idx]
    dose_va = None if dose is None else dose[val_idx]
    train = PerturbationDataset(X[train_idx], pert[train_idx], dose_tr, control_index)
    val = PerturbationDataset(X[val_idx], pert[val_idx], dose_va, control_index)
    return DatasetBundle(
        train=train,
        val=val,
        n_genes=int(X.shape[1]),
        num_perturbations=num_perturbations,
        control_index=control_index,
        meta=meta,
    )


def make_dataloaders(
    bundle: DatasetBundle, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """Standard shuffled-train / sequential-val dataloaders."""
    train_loader = DataLoader(
        bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
