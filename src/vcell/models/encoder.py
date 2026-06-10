"""MLP encoder / decoder building blocks for the VirtualCell VAE."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def build_mlp(
    in_dim: int, hidden_dims: Sequence[int], dropout: float
) -> tuple[nn.Sequential, int]:
    """Build a LayerNorm-ReLU-Dropout MLP trunk; return (module, out_dim)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers), prev


class Encoder(nn.Module):
    """Map an expression profile to a Gaussian latent (mu, logvar)."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk, last = build_mlp(n_genes, hidden_dims, dropout)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Map a latent vector back to expression space."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk, last = build_mlp(latent_dim, hidden_dims, dropout)
        self.fc_out = nn.Linear(last, n_genes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.trunk(z))
