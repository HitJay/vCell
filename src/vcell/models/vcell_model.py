"""The VirtualCell model: a latent-additive conditional VAE.

Pipeline
--------
1. Encode an expression profile to a Gaussian latent ``(mu, logvar)`` — the
   *basal* (perturbation-free) cell state.
2. Look up the perturbation's latent offset ``p`` (scaled by dose).
3. Decode ``z_basal + p`` back to expression space.

Because the decoder always receives ``basal + perturbation``, the encoder is
pushed to represent the basal state, enabling counterfactual prediction via
:meth:`VirtualCell.predict`.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vcell.models.encoder import Decoder, Encoder
from vcell.models.perturbation import PerturbationEncoder
from vcell.utils.config import ModelConfig


class VirtualCell(nn.Module):
    def __init__(
        self,
        n_genes: int,
        num_perturbations: int,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        control_index: int = 0,
        dose_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.num_perturbations = num_perturbations
        self.latent_dim = latent_dim
        self.control_index = control_index

        hidden_dims = tuple(hidden_dims)
        self.encoder = Encoder(n_genes, latent_dim, hidden_dims, dropout)
        self.decoder = Decoder(n_genes, latent_dim, tuple(reversed(hidden_dims)), dropout)
        self.pert_encoder = PerturbationEncoder(
            num_perturbations, latent_dim, control_index, dose_scaling
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        n_genes: int,
        num_perturbations: int,
        control_index: int = 0,
    ) -> "VirtualCell":
        return cls(
            n_genes=n_genes,
            num_perturbations=num_perturbations,
            latent_dim=model_cfg.latent_dim,
            hidden_dims=tuple(model_cfg.hidden_dims),
            dropout=model_cfg.dropout,
            control_index=control_index,
            dose_scaling=model_cfg.dose_scaling,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode_basal(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the basal latent ``(mu, logvar)`` for an expression profile."""
        return self.encoder(x)

    def forward(
        self,
        expression: torch.Tensor,
        pert: torch.Tensor,
        dose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(expression)
        # Sample only while training; use the mean at eval for stable outputs.
        z0 = self.reparameterize(mu, logvar) if self.training else mu
        shift = self.pert_encoder(pert, dose)
        z = z0 + shift
        x_hat = self.decoder(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def predict(
        self,
        control_expression: torch.Tensor,
        pert: int | torch.Tensor,
        dose: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Counterfactual: predict ``control_expression`` under ``pert``.

        Uses the basal mean (no sampling) so predictions are deterministic.
        """
        was_training = self.training
        self.eval()
        device = control_expression.device
        batch = control_expression.shape[0]

        mu, _ = self.encoder(control_expression)
        if not torch.is_tensor(pert):
            pert = torch.full((batch,), int(pert), dtype=torch.long, device=device)
        if dose is None:
            dose_t: torch.Tensor | None = torch.ones(batch, device=device)
        elif torch.is_tensor(dose):
            dose_t = dose.to(device)
        else:
            dose_t = torch.full((batch,), float(dose), device=device)

        shift = self.pert_encoder(pert.to(device), dose_t)
        x_hat = self.decoder(mu + shift)

        if was_training:
            self.train()
        return x_hat
