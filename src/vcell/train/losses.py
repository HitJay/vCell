"""Loss functions for the VirtualCell VAE.

Both reconstruction and KL terms are summed over features/latent dims and
averaged over the batch, so the ``beta`` weight has a consistent meaning
regardless of gene count.
"""
from __future__ import annotations

import torch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, var) || N(0, I)), summed over latent, averaged over batch."""
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()


def reconstruction_loss(x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian (MSE) reconstruction, summed over genes, averaged over batch."""
    return ((x_hat - target) ** 2).sum(dim=1).mean()


def vae_loss(
    expression: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined ELBO-style loss: reconstruction + beta * KL."""
    recon = reconstruction_loss(x_hat, expression)
    kld = kl_divergence(mu, logvar)
    total = recon + beta * kld
    return {"loss": total, "recon": recon, "kl": kld}
