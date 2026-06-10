"""Typed, YAML-backed configuration objects for vCell.

This module is intentionally free of heavy imports (no ``torch``) so the CLI can
load and manipulate configs cheaply.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _only_known(cls: type, d: dict[str, Any] | None) -> dict[str, Any]:
    """Drop keys that are not fields of ``cls`` (tolerates extra YAML keys)."""
    if not d:
        return {}
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class DataConfig:
    """Where the data comes from and how it is batched."""

    data_path: str | None = None          # .npz / .h5ad path; None -> synthetic
    n_genes: int = 200                    # synthetic / sanity dimension
    num_perturbations: int = 16           # index 0 is reserved for control
    val_fraction: float = 0.15
    batch_size: int = 128
    num_workers: int = 0
    pert_key: str = "perturbation"        # AnnData .obs column for labels
    # synthetic-only knobs (ignored when data_path is provided)
    n_cells_per_pert: int = 300
    effect_sparsity: float = 0.1
    noise_std: float = 0.3
    seed: int = 0


@dataclass
class ModelConfig:
    """Architecture hyper-parameters."""

    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    dose_scaling: bool = True


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping."""

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    kl_beta: float = 0.5
    grad_clip: float = 5.0
    patience: int = 10
    device: str = "auto"                  # auto | cpu | cuda
    out_dir: str = "runs/default"
    seed: int = 0
    log_every: int = 10


@dataclass
class Config:
    """Top-level container bundling the three config sections."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Config":
        d = d or {}
        return cls(
            data=DataConfig(**_only_known(DataConfig, d.get("data"))),
            model=ModelConfig(**_only_known(ModelConfig, d.get("model"))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train"))),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # -- overrides ------------------------------------------------------------
    def apply_overrides(self, overrides: list[str] | None) -> "Config":
        """Apply ``section.key=value`` overrides parsed as YAML scalars.

        Example: ``["train.epochs=100", "model.latent_dim=64"]``.
        """
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"Override must be 'key=value', got: {ov!r}")
            key, raw = ov.split("=", 1)
            value = yaml.safe_load(raw)
            parts = key.strip().split(".")
            obj: Any = self
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    raise KeyError(f"Unknown config section: {part!r} in {key!r}")
                obj = getattr(obj, part)
            leaf = parts[-1]
            if not hasattr(obj, leaf):
                raise KeyError(f"Unknown config key: {key!r}")
            setattr(obj, leaf, value)
        return self
