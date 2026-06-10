import torch

from vcell.models.vcell_model import VirtualCell


def _make_model() -> VirtualCell:
    return VirtualCell(
        n_genes=30,
        num_perturbations=6,
        latent_dim=8,
        hidden_dims=(32, 16),
        dropout=0.0,
    )


def test_forward_shapes():
    model = _make_model().train()
    x = torch.randn(12, 30)
    pert = torch.randint(0, 6, (12,))
    dose = torch.ones(12)
    out = model(x, pert, dose)
    assert out["x_hat"].shape == (12, 30)
    assert out["mu"].shape == (12, 8)
    assert out["logvar"].shape == (12, 8)


def test_predict_shape_and_determinism():
    model = _make_model().eval()
    ctrl = torch.randn(7, 30)
    p1 = model.predict(ctrl, 2)
    p2 = model.predict(ctrl, 2)
    assert p1.shape == (7, 30)
    assert torch.allclose(p1, p2)  # mean used -> deterministic


def test_control_perturbation_is_zero_shift():
    model = _make_model().eval()
    ctrl = torch.randn(5, 30)
    pred_control = model.predict(ctrl, 0)  # control index
    mu, _ = model.encode_basal(ctrl)
    direct = model.decoder(mu)
    assert torch.allclose(pred_control, direct, atol=1e-6)


def test_dose_scales_latent_shift_linearly():
    # Dose scales the perturbation offset linearly in *latent* space. (The
    # decoder is non-linear, so the decoded output delta is not linear in dose.)
    model = _make_model().eval()
    pert = torch.full((4,), 3, dtype=torch.long)
    shift1 = model.pert_encoder(pert, torch.ones(4))
    shift2 = model.pert_encoder(pert, torch.full((4,), 2.0))
    assert torch.allclose(shift2, 2.0 * shift1, atol=1e-6)


def test_backward_produces_gradients():
    model = _make_model().train()
    x = torch.randn(8, 30)
    pert = torch.randint(1, 6, (8,))
    dose = torch.ones(8)
    out = model(x, pert, dose)
    loss = ((out["x_hat"] - x) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
