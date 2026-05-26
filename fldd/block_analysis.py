"""Utilities for E3: analyze learned block-level joint distributions.

Given the block-factorized reverse model's joint probs p_theta(z_s^G | z_t)
of shape (B, K^|G|, Hb, Wb), we compute:
- Per-pixel marginals by marginalizing the joint over the other pixels.
- The product-of-marginals reference distribution (factorized form).
- Within-block TC = KL(joint || product-of-marginals), per block.
- Block category from a clean image x: background / mixed / stroke.
"""

import torch

from fldd.blocks import pixels_to_blocks


def _pixel_bit_table(block_size, device):
    """For each (pixel, state) pair, the value of that pixel in that state.

    Pixel ordering matches `pixels_to_blocks` / `block_indices_to_pixels`:
    state = p_0 * 2^(|G|-1) + p_1 * 2^(|G|-2) + ... + p_{|G|-1}.

    Returns: (block_size, n_states) float tensor with values in {0, 1}.
    """
    n_states = 2 ** block_size
    state_idx = torch.arange(n_states, device=device)
    bits = torch.zeros(block_size, n_states, device=device)
    for j in range(block_size):
        bit_pos = block_size - 1 - j
        bits[j] = ((state_idx >> bit_pos) & 1).float()
    return bits


def joint_to_pixel_marginals(joint_probs, block_size):
    """Marginalize a joint over the block to per-pixel Bernoulli(=1) probs.

    Args:
        joint_probs: (B, n_states, Hb, Wb), n_states = 2^block_size
        block_size: |G|
    Returns:
        marginals: (B, block_size, Hb, Wb), P(pixel_j = 1)
    """
    bits = _pixel_bit_table(block_size, joint_probs.device)  # (|G|, n_states)
    return torch.einsum("bshw,js->bjhw", joint_probs, bits)


def factorize_from_marginals(marginals, block_size, eps=1e-12):
    """Build the product-of-marginals distribution over 2^|G| joint states.

    Args:
        marginals: (B, block_size, Hb, Wb), P(pixel_j = 1)
        block_size: |G|
    Returns:
        product: (B, n_states, Hb, Wb)
    """
    bits = _pixel_bit_table(block_size, marginals.device)  # (|G|, n_states)
    log_p = torch.log(marginals.clamp(min=eps))
    log_q = torch.log((1.0 - marginals).clamp(min=eps))
    # product_log[b, s, h, w] = sum_j [ bits[j,s] * log_p + (1-bits[j,s]) * log_q ]
    product_log = (
        torch.einsum("js,bjhw->bshw", bits, log_p)
        + torch.einsum("js,bjhw->bshw", 1.0 - bits, log_q)
    )
    return product_log.exp()


def within_block_tc(joint_probs, block_size, eps=1e-12):
    """KL( joint || product-of-marginals ), per block, in nats.

    Equivalent to the total correlation of the model's per-block distribution.
    Identically 0 when block_size = 1.

    Args:
        joint_probs: (B, n_states, Hb, Wb)
    Returns:
        tc: (B, Hb, Wb)
    """
    marginals = joint_to_pixel_marginals(joint_probs, block_size)
    product = factorize_from_marginals(marginals, block_size, eps=eps)
    log_joint = torch.log(joint_probs.clamp(min=eps))
    log_product = torch.log(product.clamp(min=eps))
    return (joint_probs * (log_joint - log_product)).sum(dim=1)


def classify_blocks(x, block_size):
    """Categorize each block of x.

    0 = background: all pixels in block are 0
    2 = stroke:     all pixels in block are 1
    1 = mixed:      otherwise (i.e., a stroke boundary inside the block)

    Args:
        x: (B, 1, H, W), binary
        block_size: |G|
    Returns:
        category: (B, Hb, Wb), values in {0, 1, 2}
    """
    n_states = 2 ** block_size
    indices = pixels_to_blocks(x, block_size).squeeze(1)  # (B, Hb, Wb)
    cat = torch.ones_like(indices)        # default = mixed
    cat[indices == 0] = 0                 # background
    cat[indices == n_states - 1] = 2      # stroke
    return cat
