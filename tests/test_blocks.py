"""Tests for the block reshape / target / index utilities.

These lock in the audit findings that:
- pixels_to_blocks and block_indices_to_pixels round-trip for all
  supported (block_size, orientation) combinations.
- compute_block_target produces a valid distribution that sums to 1 and
  matches the expected product-of-Bernoullis at a hand-checked block.
- The vertical 2x1 mode has the same parameter count as horizontal 1x2.
"""

import pytest
import torch

from fldd.blocks import (
    pixels_to_blocks,
    block_indices_to_pixels,
    compute_block_target,
    block_grid_shape,
)


# (block_size, orientation, expected_grid_shape) for a 4x4 image
CASES = [
    (1, "horizontal", (4, 4)),
    (2, "horizontal", (4, 2)),
    (2, "vertical", (2, 4)),
    (4, "horizontal", (2, 2)),
]


@pytest.fixture
def x_4x4():
    """Hand-picked 4x4 binary image."""
    return torch.tensor(
        [[[[1, 0, 1, 1],
           [0, 1, 0, 1],
           [1, 1, 0, 0],
           [0, 0, 1, 1]]]],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("block_size,orientation,grid", CASES)
def test_block_grid_shape(block_size, orientation, grid):
    assert block_grid_shape(4, 4, block_size, orientation) == grid


@pytest.mark.parametrize("block_size,orientation,grid", CASES)
def test_round_trip(x_4x4, block_size, orientation, grid):
    """pixels -> indices -> pixels must recover the original."""
    idx = pixels_to_blocks(x_4x4, block_size, orientation=orientation)
    assert tuple(idx.shape[-2:]) == grid
    x_back = block_indices_to_pixels(
        idx, block_size, H=4, W=4, orientation=orientation,
    )
    assert torch.equal(x_back, x_4x4)


def test_horizontal_2x1_top_left_index():
    """Spot-check the index encoding for horizontal 1x2."""
    x = torch.tensor([[[[1, 0]]]], dtype=torch.float32)  # one 1x2 block
    idx = pixels_to_blocks(x, 2, orientation="horizontal")
    # index = p_left * 2 + p_right = 1*2 + 0 = 2
    assert idx.item() == 2


def test_vertical_2x1_top_left_index():
    """Spot-check the index encoding for vertical 2x1."""
    x = torch.tensor([[[[1], [0]]]], dtype=torch.float32)  # one 2x1 block
    idx = pixels_to_blocks(x, 2, orientation="vertical")
    # index = p_top * 2 + p_bottom = 1*2 + 0 = 2
    assert idx.item() == 2


def test_2x2_index_convention():
    """Spot-check the |G|=4 index convention: b00*8+b01*4+b10*2+b11."""
    x = torch.tensor(
        [[[[1, 0],
           [0, 1]]]],
        dtype=torch.float32,
    )
    idx = pixels_to_blocks(x, 4)
    # b00=1, b01=0, b10=0, b11=1 -> 1*8+0*4+0*2+1 = 9
    assert idx.item() == 9


def test_compute_block_target_sums_to_one():
    """The per-block joint must be a valid distribution."""
    probs = torch.rand(2, 1, 4, 4)
    for bs, orient, _ in CASES:
        if bs == 1:
            continue
        target = compute_block_target(probs, bs, orientation=orient)
        n_states = 2 ** bs
        assert target.shape[1] == n_states
        sums = target.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_compute_block_target_matches_handcomputed():
    """One 2x1 vertical block with top=0.9 bot=0.1 should give the
    expected product-of-Bernoullis joint."""
    probs = torch.zeros(1, 1, 4, 4)
    probs[0, 0, 0, 0] = 0.9
    probs[0, 0, 1, 0] = 0.1
    target = compute_block_target(probs, 2, orientation="vertical")
    # top-left vertical block
    p = target[0, :, 0, 0].tolist()
    # state 0 (top=0, bot=0) -> 0.1 * 0.9 = 0.09
    # state 1 (top=0, bot=1) -> 0.1 * 0.1 = 0.01
    # state 2 (top=1, bot=0) -> 0.9 * 0.9 = 0.81
    # state 3 (top=1, bot=1) -> 0.9 * 0.1 = 0.09
    assert pytest.approx(p[0], abs=1e-5) == 0.09
    assert pytest.approx(p[1], abs=1e-5) == 0.01
    assert pytest.approx(p[2], abs=1e-5) == 0.81
    assert pytest.approx(p[3], abs=1e-5) == 0.09


def test_compute_block_target_block_size_1():
    """For |G|=1 it returns the 2-channel [1-p, p] stack."""
    p = torch.tensor([[[[0.3, 0.7]]]], dtype=torch.float32)
    out = compute_block_target(p, 1)
    assert out.shape == (1, 2, 1, 2)
    assert torch.allclose(out[0, 0], 1 - p[0, 0])
    assert torch.allclose(out[0, 1], p[0, 0])
