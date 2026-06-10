"""Synthetic perturbation data for end-to-end testing without external files.

The generator builds a simple but non-trivial ground truth: every gene has a
baseline mean, and each (non-control) perturbation adds a *sparse* effect
vector. Cells are drawn as Gaussian noise around ``baseline + effect``. Because
the true effects are returned alongside the cells, downstream evaluation can be
sanity-checked against a known answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def generate_synthetic(
    n_genes: int = 200,
    num_perturbations: int = 16,
    n_cells_per_pert: int = 300,
    effect_sparsity: float = 0.1,
    noise_std: float = 0.3,
    control_index: int = 0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate a synthetic perturbation dataset.

    Returns a dict with ``X`` (cells x genes), ``pert`` (int ids, ``0`` =
    control), ``dose`` (all ones), plus the ground-truth ``effects`` and
    ``base_mean`` and the integer metadata used by the loaders.
    """
    if num_perturbations < 2:
        raise ValueError("num_perturbations must be >= 2 (control + >=1 pert).")
    rng = np.random.default_rng(seed)

    base_mean = rng.normal(0.0, 1.0, size=n_genes).astype(np.float32)

    effects = np.zeros((num_perturbations, n_genes), dtype=np.float32)
    n_affected = max(1, int(round(effect_sparsity * n_genes)))
    for k in range(num_perturbations):
        if k == control_index:
            continue
        idx = rng.choice(n_genes, size=n_affected, replace=False)
        effects[k, idx] = rng.normal(0.0, 1.5, size=n_affected).astype(np.float32)

    blocks_x: list[np.ndarray] = []
    blocks_p: list[np.ndarray] = []
    for k in range(num_perturbations):
        mean_k = base_mean + effects[k]
        cells = rng.normal(
            loc=mean_k, scale=noise_std, size=(n_cells_per_pert, n_genes)
        ).astype(np.float32)
        blocks_x.append(cells)
        blocks_p.append(np.full(n_cells_per_pert, k, dtype=np.int64))

    X = np.concatenate(blocks_x, axis=0)
    pert = np.concatenate(blocks_p, axis=0)
    dose = np.ones(X.shape[0], dtype=np.float32)

    order = rng.permutation(X.shape[0])
    return {
        "X": X[order],
        "pert": pert[order],
        "dose": dose[order],
        "effects": effects,
        "base_mean": base_mean,
        "control_index": np.int64(control_index),
        "num_perturbations": np.int64(num_perturbations),
        "n_genes": np.int64(n_genes),
    }


def save_npz(data: dict[str, np.ndarray], path: str | Path) -> Path:
    """Persist a dataset dict to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def generate_and_save(path: str | Path, **kwargs) -> Path:
    """Convenience: generate a synthetic dataset and write it to ``path``."""
    return save_npz(generate_synthetic(**kwargs), path)
