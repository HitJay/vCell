import numpy as np

from vcell.data.dataset import build_datasets, make_dataloaders
from vcell.data.synthetic import generate_synthetic
from vcell.utils.config import DataConfig


def test_synthetic_shapes_and_dtypes():
    d = generate_synthetic(n_genes=50, num_perturbations=5, n_cells_per_pert=20, seed=0)
    assert d["X"].shape == (5 * 20, 50)
    assert d["pert"].shape == (100,)
    assert d["dose"].shape == (100,)
    assert d["X"].dtype == np.float32
    assert d["pert"].dtype == np.int64
    assert d["effects"].shape == (5, 50)


def test_control_has_zero_effect():
    d = generate_synthetic(n_genes=40, num_perturbations=4, n_cells_per_pert=10, seed=1)
    ci = int(d["control_index"])
    assert np.allclose(d["effects"][ci], 0.0)
    assert np.any(d["effects"][1:] != 0.0)  # non-control perts do something
    assert set(np.unique(d["pert"])) == set(range(4))


def test_build_datasets_and_loaders():
    cfg = DataConfig(
        data_path=None,
        n_genes=30,
        num_perturbations=4,
        n_cells_per_pert=25,
        val_fraction=0.2,
        batch_size=16,
        seed=0,
    )
    bundle = build_datasets(cfg)
    total = len(bundle.train) + len(bundle.val)
    assert total == 4 * 25
    assert bundle.n_genes == 30
    assert bundle.num_perturbations == 4
    assert bundle.control_index == 0

    train_loader, val_loader = make_dataloaders(bundle, batch_size=16)
    batch = next(iter(train_loader))
    assert batch["expression"].shape[1] == 30
    assert batch["pert"].ndim == 1
    assert batch["dose"].ndim == 1


def test_npz_roundtrip(tmp_path):
    from vcell.data.synthetic import save_npz

    d = generate_synthetic(n_genes=20, num_perturbations=3, n_cells_per_pert=10, seed=2)
    path = save_npz(d, tmp_path / "ds.npz")
    cfg = DataConfig(data_path=str(path), val_fraction=0.25, seed=0)
    bundle = build_datasets(cfg)
    assert bundle.n_genes == 20
    assert bundle.num_perturbations == 3
