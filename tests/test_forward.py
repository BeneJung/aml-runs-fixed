"""Tests for the learned forward process (B1 + backward compatibility)."""

import math

import pytest
import torch

from fldd.forward import LearnedForwardProcess


def test_historical_alpha_floor():
    """Default sigmoid_offset = -2.0 gives floor ~ 0.0596."""
    fp = LearnedForwardProcess(T=4)
    expected = 0.5 / (1.0 + math.exp(2.0))
    assert fp.alpha_floor == pytest.approx(expected, abs=1e-8)
    assert fp.alpha_floor == pytest.approx(0.05960098, abs=1e-6)


def test_lower_offset_lowers_floor():
    """Smaller (more negative) offset must give a smaller floor."""
    fp_old = LearnedForwardProcess(T=4, sigmoid_offset=-2.0)
    fp_new = LearnedForwardProcess(T=4, sigmoid_offset=-6.0)
    assert fp_new.alpha_floor < fp_old.alpha_floor
    assert fp_new.alpha_floor == pytest.approx(
        0.5 / (1.0 + math.exp(6.0)), abs=1e-8
    )


def test_alphas_monotone_increasing():
    """At init (logits=0), alphas should be monotone-increasing in t."""
    fp = LearnedForwardProcess(T=8, sigmoid_offset=-2.0)
    alphas = fp.get_alphas()
    assert torch.all(alphas[1:] > alphas[:-1])


def test_alphas_strictly_below_half():
    """Parameterization upper bound is 0.5; alphas must lie in (floor, 0.5)."""
    fp = LearnedForwardProcess(T=4, sigmoid_offset=-2.0)
    alphas = fp.get_alphas()
    assert (alphas > fp.alpha_floor - 1e-6).all()
    assert (alphas < 0.5).all()


def test_old_ckpt_loads_strict():
    """Old-style ckpt (just `logits`) must load with strict=True so the
    historical results reproduce."""
    fp = LearnedForwardProcess(T=4)
    fp.load_state_dict({"logits": torch.tensor([0.5, 0.8, 1.0, 1.2])},
                       strict=True)
    alphas = fp.get_alphas()
    assert alphas.shape == (4,)


def test_fixed_alphas_ignores_logits():
    """In fixed-alphas mode the schedule must ignore the (non-trainable)
    logits — even if we manually perturb them, get_alphas is unchanged."""
    target = [0.06, 0.5]
    fp = LearnedForwardProcess(T=2, fixed_alphas=target)
    before = fp.get_alphas().detach().clone()

    # Manually perturb the (frozen) logits. In fixed mode this must be a no-op.
    with torch.no_grad():
        fp.logits.add_(torch.tensor([5.0, -5.0]))
    after = fp.get_alphas().detach().clone()
    assert torch.allclose(before, after, atol=1e-7)


def test_fixed_alphas_logits_have_no_grad():
    """The logits parameter in fixed mode must have requires_grad=False."""
    fp = LearnedForwardProcess(T=2, fixed_alphas=[0.1, 0.4])
    assert fp.logits.requires_grad is False


def test_fixed_alphas_value_validation():
    """fixed_alphas must satisfy length and range constraints."""
    with pytest.raises(ValueError, match="shape"):
        LearnedForwardProcess(T=4, fixed_alphas=[0.1, 0.2])
    with pytest.raises(ValueError, match="0.5"):
        LearnedForwardProcess(T=2, fixed_alphas=[0.1, 0.51])
    with pytest.raises(ValueError, match="0.5"):
        LearnedForwardProcess(T=2, fixed_alphas=[0.0, 0.4])
    # boundary cases that should be OK
    LearnedForwardProcess(T=2, fixed_alphas=[0.05, 0.5])  # 0.5 allowed


def test_sample_zt_returns_binary():
    """sample_zt must return {0, 1}-valued tensor matching x's shape."""
    fp = LearnedForwardProcess(T=4)
    x = torch.bernoulli(0.5 * torch.ones(2, 1, 8, 8))
    z, p = fp.sample_zt(x, t_idx=2)
    assert z.shape == x.shape
    assert torch.all((z == 0) | (z == 1))
    assert (p >= 0).all() and (p <= 1).all()


def test_kl_prior_zero_when_alpha_T_is_half():
    """With α_T = 0.5, q(z_T|x) = Bern(0.5) = prior, so KL = 0."""
    fp = LearnedForwardProcess(T=2, fixed_alphas=[0.1, 0.5])
    x = torch.bernoulli(0.5 * torch.ones(4, 1, 8, 8))
    kl = fp.kl_prior(x).item()
    assert abs(kl) < 1e-5
