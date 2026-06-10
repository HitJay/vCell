"""Perturbation embeddings composed additively in latent space.

Each perturbation maps to a learned latent vector. The control perturbation
always contributes a zero shift, so control cells are reconstructed from their
basal latent unchanged. An optional dose multiplier scales the shift linearly.
"""
from __future__ import annotations

import torch
from torch import nn


class PerturbationEncoder(nn.Module):
    """Embed perturbation ids into latent-space offset vectors."""

    def __init__(
        self,
        num_perturbations: int,
        latent_dim: int,
        control_index: int = 0,
        dose_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.num_perturbations = num_perturbations
        self.control_index = control_index
        self.dose_scaling = dose_scaling
        self.embedding = nn.Embedding(num_perturbations, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        with torch.no_grad():
            self.embedding.weight[control_index].zero_()

    def forward(
        self, pert_idx: torch.Tensor, dose: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = self.embedding(pert_idx)                       # (B, latent_dim)
        if self.dose_scaling and dose is not None:
            p = p * dose.unsqueeze(-1)
        # The control perturbation never shifts the latent state.
        keep = (pert_idx != self.control_index).unsqueeze(-1).to(p.dtype)
        return p * keep
