"""Requirement #6 — the 'simple / statistical' baseline for binarized MNIST.

The brief asks for a simple statistical baseline ("sampling from the training
label distribution"). For a *generative* task on binarized MNIST the natural
analog is: sample each pixel independently from its training-set Bernoulli
marginal — i.e. a model that knows the per-pixel ink frequency but nothing
about how pixels relate. We report two variants:

  * per_pixel : z_i ~ Bernoulli(p_i),  p_i = mean over train of pixel i  (1×28×28)
  * global    : z_i ~ Bernoulli(p),    p   = global ink fraction (one scalar)

Both are training-free and take ~1 min. FID is scored with the SAME in-memory
InceptionV3 protocol as run_e2_fast.py / fid_no_ema.py, so the number drops
straight into the comparison table next to |G|=1 and |G|=4.

Usage (Renku):
    cd ~/work/aml-runs-fixed
    python scripts/baseline_simple.py --n_fid_samples 10000 \
        --out results/v3/baseline_simple.json
"""

import argparse
import json
import os

import numpy as np
import torch

# Make `fldd` importable when run as `python scripts/baseline_simple.py`.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fldd.data import get_binarized_mnist
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


def features(x_1ch, inc, device, batch_size=50):
    fs = []
    with torch.no_grad():
        for i in range(0, x_1ch.shape[0], batch_size):
            x = x_1ch[i:i + batch_size].to(device).repeat(1, 3, 1, 1)
            f = inc(x)[0].squeeze(3).squeeze(2)
            fs.append(f.cpu().numpy())
    return np.concatenate(fs, axis=0)


def fid_of(samples_1ch, inc, mu_r, sig_r, device):
    gf = features(samples_1ch, inc, device)
    return float(calculate_frechet_distance(mu_r, sig_r, gf.mean(0),
                                            np.cov(gf, rowvar=False)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_fid_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/v3/baseline_simple.json")
    args = p.parse_args()
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    N = args.n_fid_samples

    # Load all of train (for marginals) and N test images (FID reference).
    train_loader, test_loader = get_binarized_mnist(batch_size=4096)
    train = torch.cat([x for (x,) in train_loader], dim=0)          # (60000,1,28,28)
    for (x,) in get_binarized_mnist(batch_size=N)[1]:
        real = x[:N]; break

    p_pixel = train.mean(dim=0, keepdim=True)        # (1,1,28,28) per-pixel rate
    p_global = float(train.mean())                   # scalar ink fraction
    print(f"train ink fraction (global) = {p_global:.4f}; "
          f"per-pixel range [{p_pixel.min():.3f}, {p_pixel.max():.3f}]")

    # Draw independent-Bernoulli samples.
    per_pixel = torch.bernoulli(p_pixel.expand(N, 1, 28, 28), generator=g)
    glob = torch.bernoulli(torch.full((N, 1, 28, 28), p_global), generator=g)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc = InceptionV3([block_idx]).to(args.device).eval()
    print(f"computing real features {tuple(real.shape)}...")
    rf = features(real, inc, args.device)
    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)

    fid_pp = fid_of(per_pixel, inc, mu_r, sig_r, args.device)
    fid_gl = fid_of(glob, inc, mu_r, sig_r, args.device)

    out = {
        "config": {"n_fid_samples": N, "seed": args.seed,
                   "note": "training-free independent-pixel Bernoulli baselines"},
        "global_rate": {"p": p_global, "fid": fid_gl},
        "per_pixel_marginal": {"fid": fid_pp},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n=== SIMPLE STATISTICAL BASELINE (FID @ %d) ===" % N)
    print(f"  global-rate Bernoulli     : FID = {fid_gl:.2f}")
    print(f"  per-pixel-marginal Bernoulli: FID = {fid_pp:.2f}")
    print("  (compare to |G|=1 diffusion ~46-58 and |G|=4 ~28-49 — shows how much")
    print("   structure the diffusion models add over independent pixels.)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
