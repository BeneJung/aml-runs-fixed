"""Tests for the training-loss derivations (B5: ce vs true_elbo)."""

import pytest
import torch

from fldd.forward import LearnedForwardProcess
from fldd.train import compute_elbo_loss
from fldd.unet import UNet


@pytest.fixture
def tiny_setup():
    torch.manual_seed(0)
    T = 4
    fp = LearnedForwardProcess(T=T)
    model = UNet(channels=(8, 16), block_size=1)
    x = torch.bernoulli(0.5 * torch.ones(4, 1, 28, 28))
    return model, fp, x, T


def _grad_dict(model):
    return {n: p.grad.detach().clone() if p.grad is not None else None
            for n, p in model.named_parameters()}


def test_ce_and_true_elbo_have_same_theta_gradient(tiny_setup):
    """Subtracting H[q(z_s|x)] from BCE doesn't depend on theta, so the
    gradient w.r.t. the reverse-model parameters must be identical."""
    model, fp, x, T = tiny_setup

    model.zero_grad()
    torch.manual_seed(42)
    l, _ = compute_elbo_loss(model, fp, x, T, block_size=1, loss_form="ce")
    l.backward()
    g_ce = _grad_dict(model)

    model.zero_grad()
    torch.manual_seed(42)
    l, _ = compute_elbo_loss(model, fp, x, T, block_size=1, loss_form="true_elbo")
    l.backward()
    g_elbo = _grad_dict(model)

    for k in g_ce:
        if g_ce[k] is None:
            continue
        diff = (g_ce[k] - g_elbo[k]).abs().max().item()
        assert diff < 1e-6, f"parameter {k!r}: max grad diff = {diff}"


def test_true_elbo_smaller_than_ce(tiny_setup):
    """true_elbo = ce - H[q(z_s|x)] for t > 1, and H >= 0 always, so
    the true_elbo recon must be <= the ce recon."""
    model, fp, x, T = tiny_setup
    torch.manual_seed(0)
    _, m_ce = compute_elbo_loss(model, fp, x, T, block_size=1, loss_form="ce")
    torch.manual_seed(0)
    _, m_elbo = compute_elbo_loss(model, fp, x, T, block_size=1, loss_form="true_elbo")
    assert m_elbo["recon"] <= m_ce["recon"] + 1e-4


def test_loss_form_validation(tiny_setup):
    model, fp, x, T = tiny_setup
    with pytest.raises(ValueError):
        compute_elbo_loss(model, fp, x, T, block_size=1, loss_form="bogus")


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_loss_runs_without_error(block_size):
    """All block sizes must produce a finite loss + finite gradient."""
    torch.manual_seed(0)
    fp = LearnedForwardProcess(T=2)
    model = UNet(channels=(8, 16), block_size=block_size)
    x = torch.bernoulli(0.5 * torch.ones(2, 1, 28, 28))
    loss, m = compute_elbo_loss(model, fp, x, 2, block_size=block_size,
                                loss_form="true_elbo")
    assert torch.isfinite(loss)
    loss.backward()
    # at least one parameter must have a non-zero finite gradient
    g_any_finite = False
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            g_any_finite = True
            break
    assert g_any_finite


def test_vertical_block_loss_runs():
    """The 2x1 vertical orientation must train end-to-end."""
    torch.manual_seed(0)
    fp = LearnedForwardProcess(T=2)
    model = UNet(channels=(8, 16), block_size=2, orientation="vertical")
    x = torch.bernoulli(0.5 * torch.ones(2, 1, 28, 28))
    loss, m = compute_elbo_loss(
        model, fp, x, 2, block_size=2, loss_form="true_elbo",
        orientation="vertical",
    )
    assert torch.isfinite(loss)
