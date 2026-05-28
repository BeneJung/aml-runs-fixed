"""Pytest checks on the LearnedForwardProcess parameterization.

Generalized from the teammate's `sanity_schedule.py`. We test BOTH the
historical sigmoid_offset=-2 (which should hit the floor at ~0.0596) AND
the v2 fix sigmoid_offset=-6 (which should NOT hit a floor in practice and
should be free to reach near-zero alpha values).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from fldd.forward import LearnedForwardProcess


T_VALUES = [4]


def _floor(offset, T):
    """Smallest reachable alpha_1 given offset."""
    return 0.5 / (1.0 + math.exp(-offset))


def _drive_schedule_to_target(offset, target_val, T=4, steps=2000, lr=0.1, seed=0):
    """Optimize the logits to minimize ((alphas - target_val)**2).mean().

    Returns the final alpha vector after `steps` Adam steps.
    """
    torch.manual_seed(seed)
    fp = LearnedForwardProcess(T=T, sigmoid_offset=offset)
    opt = torch.optim.Adam(fp.parameters(), lr=lr)
    target = torch.full((T,), float(target_val))
    for _ in range(steps):
        opt.zero_grad()
        loss = ((fp.get_alphas() - target) ** 2).mean()
        loss.backward()
        opt.step()
    return fp.get_alphas().detach().cpu()


@pytest.mark.parametrize("T", T_VALUES)
def test_init_alphas_monotone(T):
    """At logits=0 the schedule should be monotone non-decreasing."""
    fp = LearnedForwardProcess(T=T, sigmoid_offset=-6.0)
    a = fp.get_alphas().detach()
    assert torch.all(a[1:] >= a[:-1]), f"init alphas not monotone: {a.tolist()}"


@pytest.mark.parametrize("T", T_VALUES)
def test_init_alphas_within_bounds(T):
    """All alphas must lie strictly in (0, 0.5]."""
    for offset in (-2.0, -6.0):
        fp = LearnedForwardProcess(T=T, sigmoid_offset=offset)
        a = fp.get_alphas().detach()
        assert (a > 0).all(), f"alpha not > 0 at offset={offset}: {a.tolist()}"
        assert (a <= 0.5).all(), f"alpha > 0.5 at offset={offset}: {a.tolist()}"


@pytest.mark.parametrize("T", T_VALUES)
def test_floor_at_offset_minus_2(T):
    """With offset=-2 the schedule cannot go below ~0.0596 — historical floor."""
    expected_floor = _floor(-2.0, T)
    final = _drive_schedule_to_target(-2.0, target_val=0.0, T=T, steps=3000)
    # All driven-down alphas should pin within ~1e-3 of the floor
    assert torch.allclose(final, torch.full_like(final, expected_floor), atol=1e-3), \
        f"offset=-2 driven-down alphas: {final.tolist()} vs floor={expected_floor:.4f}"


@pytest.mark.parametrize("T", T_VALUES)
def test_no_effective_floor_at_offset_minus_6(T):
    """With offset=-6 the schedule should be able to reach near-zero alpha."""
    final = _drive_schedule_to_target(-6.0, target_val=0.0, T=T, steps=3000)
    # All driven-down alphas should be well below 0.0596 (the offset=-2 floor)
    assert (final < 0.01).all(), \
        f"offset=-6 supposed to allow alpha→0 but got {final.tolist()}"


@pytest.mark.parametrize("T", T_VALUES)
@pytest.mark.parametrize("offset", [-2.0, -6.0])
def test_can_drive_schedule_up(T, offset):
    """Both offsets must let the schedule reach the upper bound ~0.5."""
    final = _drive_schedule_to_target(offset, target_val=0.49, T=T, steps=3000)
    assert (final > 0.48).all(), \
        f"offset={offset}: schedule didn't reach 0.49: {final.tolist()}"


@pytest.mark.parametrize("T", T_VALUES)
def test_alpha_T_pins_at_upper_bound(T):
    """The 0.5 cap is structural — alpha_T cannot exceed 0.5 by construction."""
    fp = LearnedForwardProcess(T=T, sigmoid_offset=-6.0)
    # Hammer the last logit very positive
    with torch.no_grad():
        fp.logits.copy_(torch.tensor([10.0] * T))
    a = fp.get_alphas().detach()
    assert (a <= 0.5).all() and a[-1] > 0.499, \
        f"alpha_T should pin at upper bound 0.5: {a.tolist()}"


@pytest.mark.parametrize("T", T_VALUES)
def test_alpha_floor_property_matches_math(T):
    """The `alpha_floor` property must equal 0.5*sigmoid(sigmoid_offset)."""
    for offset in (-2.0, -3.5, -6.0):
        fp = LearnedForwardProcess(T=T, sigmoid_offset=offset)
        expected = 0.5 / (1.0 + math.exp(-offset))
        assert abs(fp.alpha_floor - expected) < 1e-8, \
            f"alpha_floor mismatch at offset={offset}"
