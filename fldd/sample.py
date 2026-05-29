import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from fldd.blocks import block_indices_to_pixels


@torch.no_grad()
def sample(model, forward_process, T, n_samples=64, device="cpu",
           block_size=1, orientation="horizontal", generator=None, z_init=None):
    """Generate samples by running the reverse process.

    Start from z_T ~ Uniform({0,1}) and iteratively apply
    p_theta(z_{t-1} | z_t) for t = T, T-1, ..., 1.

    Handles both pixel-factorized (block_size=1) and block-factorized models.

    Optional args (backward-compatible; default to historical behaviour):
      generator: torch.Generator for reproducible / seeded sampling. Threaded
                 through every stochastic draw (bernoulli, multinomial). Must
                 live on the same device as `device` for CUDA.
      z_init:    (n, 1, 28, 28) starting noise. If given, it is used verbatim
                 (e.g. to start |G|=1 and |G|=4 from the *same* z_T for a fair
                 qualitative comparison) and `n_samples` is taken from it.
    """
    model.eval()
    forward_process.eval()

    # start from uniform noise (or a caller-supplied z_T)
    if z_init is not None:
        z = z_init.to(device)
        n_samples = z.shape[0]
    else:
        z = torch.bernoulli(
            0.5 * torch.ones(n_samples, 1, 28, 28, device=device),
            generator=generator,
        )

    for t in range(T, 0, -1):
        t_batch = torch.full((n_samples,), t - 1, device=device, dtype=torch.long)
        logits = model(z, t_batch)

        if block_size == 1:
            probs = torch.sigmoid(logits)
            if t > 1:
                z = torch.bernoulli(probs, generator=generator)
            else:
                z = (probs > 0.5).float()
        else:
            # block-factorized: sample from categorical over block states
            # logits: (B, K^|G|, Hb, Wb)
            probs = F.softmax(logits, dim=1)

            if t > 1:
                # sample block indices from categorical
                B, n_states, Hb, Wb = probs.shape
                # reshape to (B*Hb*Wb, n_states) for multinomial
                flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, n_states)
                flat_indices = torch.multinomial(
                    flat_probs, 1, generator=generator).squeeze(-1)
                indices = flat_indices.reshape(B, 1, Hb, Wb)
            else:
                # final step: take argmax
                indices = logits.argmax(dim=1, keepdim=True)

            z = block_indices_to_pixels(indices, block_size, orientation=orientation)

    return z


def save_samples(samples, path, nrow=8):
    """Save a grid of samples as an image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(samples, path, nrow=nrow)
