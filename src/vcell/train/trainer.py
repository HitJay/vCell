"""Training loop with early stopping, checkpointing and counterfactual eval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vcell.data.dataset import DatasetBundle, build_datasets, make_dataloaders
from vcell.models.vcell_model import VirtualCell
from vcell.train.losses import vae_loss
from vcell.utils.config import Config
from vcell.utils.metrics import delta_metrics
from vcell.utils.seed import resolve_device, set_seed


class Trainer:
    """Drive optimisation of a :class:`VirtualCell` on a dataset bundle."""

    def __init__(
        self,
        model: VirtualCell,
        config: Config,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or resolve_device(config.train.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.out_dir = Path(config.train.out_dir)
        self.history: list[dict[str, float]] = []

    # -- batch / epoch --------------------------------------------------------
    def _step_batch(self, batch: dict[str, torch.Tensor], train: bool) -> dict[str, float]:
        expr = batch["expression"].to(self.device)
        pert = batch["pert"].to(self.device)
        dose = batch["dose"].to(self.device)
        out = self.model(expr, pert, dose)
        losses = vae_loss(
            expr, out["x_hat"], out["mu"], out["logvar"], beta=self.config.train.kl_beta
        )
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if self.config.train.grad_clip:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip
                )
            self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    def _run_epoch(self, loader: Any, train: bool) -> dict[str, float]:
        self.model.train(train)
        agg: dict[str, float] = {}
        n = 0
        for batch in loader:
            bs = int(batch["expression"].shape[0])
            with torch.set_grad_enabled(train):
                stats = self._step_batch(batch, train)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v * bs
            n += bs
        return {k: v / max(n, 1) for k, v in agg.items()}

    # -- fit ------------------------------------------------------------------
    def fit(self, bundle: DatasetBundle) -> list[dict[str, float]]:
        set_seed(self.config.train.seed)
        train_loader, val_loader = make_dataloaders(
            bundle, self.config.data.batch_size, self.config.data.num_workers
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(self.out_dir / "config.yaml")

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for epoch in range(1, self.config.train.epochs + 1):
            train_stats = self._run_epoch(train_loader, train=True)
            val_stats = self._run_epoch(val_loader, train=False)
            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
            }
            self.history.append(record)

            if epoch == 1 or epoch % self.config.train.log_every == 0:
                print(
                    f"[{epoch:3d}/{self.config.train.epochs}] "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"val_loss={val_stats['loss']:.4f} "
                    f"(recon={val_stats['recon']:.4f} kl={val_stats['kl']:.4f})"
                )

            if val_stats["loss"] < best_val - 1e-4:
                best_val = val_stats["loss"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                bad_epochs = 0
                self.save_checkpoint(self.out_dir / "best.ckpt", epoch, best_val)
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.train.patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best val_loss={best_val:.4f})."
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.save_history(self.out_dir / "history.json")
        return self.history

    # -- evaluation -----------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self, bundle: DatasetBundle
    ) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        """Counterfactual Δ-expression agreement, per perturbation and averaged."""
        self.model.eval()
        control = bundle.val.control_matrix()
        if control.shape[0] == 0:
            control = bundle.train.control_matrix()
        control = control.to(self.device)
        control_mean = control.mean(dim=0).cpu().numpy()

        per_pert: dict[int, dict[str, float]] = {}
        for k in bundle.val.present_perturbations():
            if k == bundle.control_index:
                continue
            true_mean_t = bundle.val.perturbation_mean(k)
            if true_mean_t is None or control.shape[0] == 0:
                continue
            true_mean = true_mean_t.cpu().numpy()
            pred_mean = self.model.predict(control, k).mean(dim=0).cpu().numpy()
            per_pert[k] = delta_metrics(
                pred_delta=pred_mean - control_mean,
                true_delta=true_mean - control_mean,
            )

        summary: dict[str, float] = {}
        if per_pert:
            metric_keys = next(iter(per_pert.values())).keys()
            for key in metric_keys:
                vals = [pp[key] for pp in per_pert.values()]
                summary[key] = float(np.nanmean(vals))
        summary["n_perturbations_evaluated"] = float(len(per_pert))
        return summary, per_pert

    # -- io -------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": self.config.to_dict(),
                "n_genes": self.model.n_genes,
                "num_perturbations": self.model.num_perturbations,
                "control_index": self.model.control_index,
            },
            path,
        )

    def save_history(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[VirtualCell, Config, dict[str, Any]]:
    """Reconstruct a :class:`VirtualCell` and its config from a checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    config = Config.from_dict(ckpt["config"])
    model = VirtualCell.from_config(
        config.model,
        n_genes=int(ckpt["n_genes"]),
        num_perturbations=int(ckpt["num_perturbations"]),
        control_index=int(ckpt["control_index"]),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, config, ckpt


def run_training(
    config: Config,
) -> tuple[VirtualCell, Trainer, dict[str, float], DatasetBundle]:
    """End-to-end: build data, train, evaluate and persist results."""
    set_seed(config.train.seed)
    bundle = build_datasets(config.data)
    model = VirtualCell.from_config(
        config.model,
        n_genes=bundle.n_genes,
        num_perturbations=bundle.num_perturbations,
        control_index=bundle.control_index,
    )
    trainer = Trainer(model, config)
    trainer.fit(bundle)
    summary, per_pert = trainer.evaluate(bundle)

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary,
                "per_perturbation": {str(k): v for k, v in per_pert.items()},
            },
            fh,
            indent=2,
        )
    print("Evaluation:", {k: round(v, 4) for k, v in summary.items()})
    return model, trainer, summary, bundle
