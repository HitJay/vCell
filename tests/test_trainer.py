import numpy as np
import torch

from vcell.train.trainer import load_checkpoint, run_training
from vcell.utils.config import Config


def _tiny_config(out_dir) -> Config:
    cfg = Config()
    cfg.data.n_genes = 40
    cfg.data.num_perturbations = 5
    cfg.data.n_cells_per_pert = 80
    cfg.data.batch_size = 32
    cfg.data.val_fraction = 0.2
    cfg.data.noise_std = 0.3
    cfg.data.seed = 0
    cfg.model.latent_dim = 8
    cfg.model.hidden_dims = [32, 16]
    cfg.model.dropout = 0.0
    cfg.train.epochs = 30
    cfg.train.patience = 100
    cfg.train.device = "cpu"
    cfg.train.out_dir = str(out_dir)
    cfg.train.log_every = 100
    cfg.train.seed = 0
    return cfg


def test_training_runs_improves_and_writes_artifacts(tmp_path):
    out_dir = tmp_path / "run"
    cfg = _tiny_config(out_dir)
    model, trainer, summary, bundle = run_training(cfg)

    assert len(trainer.history) >= 1
    first_loss = trainer.history[0]["train_loss"]
    last_loss = trainer.history[-1]["train_loss"]
    assert last_loss < first_loss  # optimisation reduces the loss

    assert summary["n_perturbations_evaluated"] >= 1
    assert np.isfinite(summary["delta_pearson"])
    # easy synthetic signal should be recovered well above chance
    assert summary["delta_pearson"] > 0.3

    assert (out_dir / "best.ckpt").exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "eval.json").exists()
    assert (out_dir / "history.json").exists()


def test_checkpoint_roundtrip(tmp_path):
    out_dir = tmp_path / "run"
    cfg = _tiny_config(out_dir)
    model, trainer, summary, bundle = run_training(cfg)

    loaded, loaded_cfg, ckpt = load_checkpoint(out_dir / "best.ckpt")
    assert loaded_cfg.model.latent_dim == cfg.model.latent_dim

    x = bundle.val.control_matrix()[:4]
    if x.shape[0] == 0:
        x = bundle.train.control_matrix()[:4]
    a = model.predict(x, 1)
    b = loaded.predict(x, 1)
    assert torch.allclose(a, b, atol=1e-5)
