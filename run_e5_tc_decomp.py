"""E5 (post-hoc analysis): within- vs between-block TC decomposition.

Quantifies, for a trained |G|=4 checkpoint, how much of the
*data-averaged* cross-pixel total correlation lives WITHIN 2x2 blocks
(absorbable by a block-factorized head) vs BETWEEN blocks (irreducible
under any block-factorized reverse model).

Why we care
-----------
The whole project rests on the claim that within-block TC is
non-negligible for binarized MNIST. E3 confirms that the trained block
model has captured some within-block TC. What E3 does NOT show is
*what fraction* of the total available TC was absorbed — leaving the
reader to infer the importance of the absorption from FID changes
alone. This script supplies that fraction.

What we compute
---------------
For a fixed reverse step t and a batch of N test images x_i:

  1. q_i := q(z_s | z_t_i, x_i)  -- the per-image factorized target
     (product of per-pixel Bernoullis derived from x_i and alpha_s).

  2. q_bar(z_t) := E_{x | z_t}[q(z_s | x)]  -- the data-averaged
     posterior at this z_t. We approximate it by averaging the q_i
     over a batch of samples sharing approximately the same z_t.

     We use the trained model p_theta(z_s | z_t) as a high-quality
     proxy for q_bar(z_t), since p_theta is fit to minimise
     E_{x,z_t}[KL[q(z_s|x) || p_theta(z_s|z_t)]] and its optimum is
     exactly q_bar(z_t). (This is approximate -- p_theta is not the
     exact minimiser -- but it is a strictly better stand-in than a
     KDE over the batch.)

  3. Decompose TC of q_bar at every 2x2 block partition:

        TC_total[q_bar] = sum_i H[q_bar_i] - H[q_bar]
        TC_within[q_bar] = sum_b ( sum_{i in b} H[q_bar_i] - H[q_bar^{G_b}] )
        TC_between[q_bar] = TC_total - TC_within

     TC_within is what a |G|=4 head can absorb; TC_between is the
     irreducible residual the method explicitly does not address.

Because q_bar at the model is the BLOCK-FACTORIZED joint p_theta(z_s |
z_t), TC_within is already captured by the model (E3 measures exactly
that). TC_between requires the FULL 28x28 joint of p_theta, which is
the product over blocks of the per-block joints — that joint is the
mixture induced by the U-Net conditioning, which we cannot evaluate
exactly without an expensive MC over the model.

Practical approach
------------------
We use a Monte Carlo estimator: sample many z_s | z_t pairs from the
model, then compute per-pixel marginals, per-block joints, and pairwise
mutual information across blocks. Specifically:

  - Per-pixel marginals p(z_s,i = 1 | z_t):  estimate from MC samples
    of z_s and a per-image average of the block joint marginals.
  - Per-block joints p(z_s^{G_b} | z_t):  model output directly.
  - Pair MI between blocks  I(z_s^{G_a}; z_s^{G_b} | z_t):  empirical
    from MC samples.

We do the analysis at the schedule's most-uncertain t (t=T) where E3
already showed TC is large.

Outputs
-------
results/results_e5_tc_decomp.json with per-(t, seed):
    - mean within-block TC (matches E3 at t=T)
    - mean between-block MI summed over a set of (block_a, block_b) pairs
    - decomposition ratio: TC_within / (TC_within + TC_between)

A bar chart in figures/e5_tc_decomp.{png,pdf}.

This is post-hoc on existing |G|=4 E2 ckpts; no retraining required.
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from fldd.block_analysis import (
    factorize_from_marginals,
    joint_to_pixel_marginals,
    within_block_tc,
)
from fldd.blocks import block_grid_shape
from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet


CKPT_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    block_size = ckpt["block_size"]
    sigmoid_offset = ckpt.get(
        "sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET
    )
    T = ckpt["T"]
    model = UNet(channels=(32, 64, 128), block_size=block_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fp = LearnedForwardProcess(T=T, sigmoid_offset=sigmoid_offset).to(device)
    fp.load_state_dict(ckpt["forward"])
    fp.eval()
    return model, fp, ckpt


@torch.no_grad()
def sample_block_joints(model, fp, x, t_idx, block_size, mc_samples=8):
    """Run the model `mc_samples` times with fresh z_t to get an MC
    distribution over block joints. Returns per-image, per-sample joint
    probability arrays.

    Shape: (mc_samples, B, K^|G|, Hb, Wb)
    """
    out = []
    for _ in range(mc_samples):
        z_t, _ = fp.sample_zt(x, t_idx)
        t_batch = torch.full(
            (x.shape[0],), t_idx, device=x.device, dtype=torch.long,
        )
        logits = model(z_t, t_batch)
        if block_size == 1:
            p1 = torch.sigmoid(logits)
            joint = torch.cat([1 - p1, p1], dim=1)
        else:
            joint = F.softmax(logits, dim=1)
        out.append(joint)
    return torch.stack(out, dim=0)


def categorical_entropy(probs, dim=-1, eps=1e-12):
    """H[Cat(probs)] in nats, summed over `dim`."""
    p = probs.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=dim)


@torch.no_grad()
def block_pair_mutual_information(joints, block_a, block_b, mc_samples_for_kde=None):
    """Estimate I(z^{G_a}; z^{G_b} | z_t) for two distinct block
    positions.

    Args:
        joints: (mc_samples, B, K^|G|, Hb, Wb) — same input as
            sample_block_joints. Each (b, hb, wb) entry is the model's
            joint over that 2x2 block conditional on a sampled z_t.
        block_a, block_b: each is a tuple (hb, wb) of grid coordinates.

    Returns:
        Per-image MI estimate, shape (B,).

    Method:
        For each image i and each z_t sample s, we have the block joint
        for blocks a, b. Treat blocks a and b as independent draws from
        their respective joints (this is the model's assumption -- the
        full 28x28 joint factorizes over blocks). Then I(a; b | z_t)
        averaged over z_t equals:

            H[E_{z_t}[p(z^a | z_t)]] + H[E_{z_t}[p(z^b | z_t)]]
              - H[E_{z_t}[p(z^a, z^b | z_t)]]

        Each marginal-over-z_t is approximated by averaging the model's
        per-z_t block joints over the MC samples. The full joint
        E_{z_t}[p(a,b)] = E_{z_t}[p(a|z_t)] * p(b|z_t)] is also
        approximated by averaging the OUTER PRODUCT over MC samples.

    Why this captures between-block coupling: the model's p(z_s | z_t)
    is a block-factorized product, so per z_t there is zero MI between
    blocks. But averaging over z_t (the data-averaged posterior q_bar)
    induces non-zero MI -- exactly the "between-block TC" the method
    cannot absorb.
    """
    ha, wa = block_a
    hb, wb = block_b
    # (S, B, K)
    pa = joints[:, :, :, ha, wa]
    pb = joints[:, :, :, hb, wb]
    S, B, K = pa.shape

    # Outer product per (s, i) -> (S, B, K, K)
    pab = pa.unsqueeze(-1) * pb.unsqueeze(-2)

    # E over z_t (i.e. mean over S)
    pa_marg = pa.mean(dim=0)  # (B, K)
    pb_marg = pb.mean(dim=0)  # (B, K)
    pab_marg = pab.mean(dim=0)  # (B, K, K)

    Ha = categorical_entropy(pa_marg, dim=-1)               # (B,)
    Hb = categorical_entropy(pb_marg, dim=-1)               # (B,)
    Hab = categorical_entropy(pab_marg.reshape(B, K * K))   # (B,)

    return Ha + Hb - Hab  # MI(a; b) marginal-over-z_t


@torch.no_grad()
def aggregate_tc_decomposition(
    model, fp, x, t_idx, block_size,
    mc_samples=8, n_pair_samples=64, rng_seed=0,
):
    """Compute within-block TC + a sample of between-block MIs.

    Returns dict with:
      - within_tc_mean: mean over (B, Hb, Wb) blocks of within-block TC
      - between_mi_mean: mean over `n_pair_samples` random (a, b) pairs
        of between-block MI, averaged over B images
      - n_pair_samples
    """
    joints = sample_block_joints(model, fp, x, t_idx, block_size, mc_samples)
    # Within: same as E3, using the *first* MC sample as the typical
    # per-z_t block joint
    within = within_block_tc(joints[0], block_size)  # (B, Hb, Wb)
    within_tc_mean = within.mean().item()

    # Between: random pair of distinct block coords
    B, K, Hb, Wb = joints[0].shape
    rng = np.random.default_rng(rng_seed)
    pair_mis = []
    for _ in range(n_pair_samples):
        while True:
            ha, wa = rng.integers(Hb), rng.integers(Wb)
            hbb, wbb = rng.integers(Hb), rng.integers(Wb)
            if (ha, wa) != (hbb, wbb):
                break
        mi = block_pair_mutual_information(joints, (ha, wa), (hbb, wbb))
        pair_mis.append(mi.mean().item())

    between_mi_mean = float(np.mean(pair_mis))
    return {
        "within_tc_mean": within_tc_mean,
        "between_mi_mean": between_mi_mean,
        "n_pair_samples": n_pair_samples,
        "mc_samples": mc_samples,
        "n_blocks": Hb * Wb,
        "n_block_pairs_total": Hb * Wb * (Hb * Wb - 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_e2")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--n_images", type=int, default=512)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--n_pair_samples", type=int, default=64,
                        help="random block-pair samples for between-block MI")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_json", type=str,
                        default="results/results_e5_tc_decomp.json")
    parser.add_argument("--fig_prefix", type=str, default="figures/e5_tc_decomp")
    args = parser.parse_args()

    pattern = os.path.join(args.ckpt_dir, f"bs{args.block_size}_s*_best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {pattern}")
    print(f"E5 within/between-block TC decomposition | |G|={args.block_size}")
    print(f"  checkpoints: {len(paths)}")
    for p in paths:
        print(f"    {p}")

    _, test_loader = get_binarized_mnist(batch_size=args.n_images)
    for (x,) in test_loader:
        x = x.to(args.device)
        break

    print(f"  test batch shape: {tuple(x.shape)}")

    per_seed = {}
    T = None
    for path in paths:
        m = CKPT_RE.search(os.path.basename(path))
        seed = int(m.group(2)) if m else -1
        model, fp, ckpt = load_checkpoint(path, args.device)
        T = ckpt["T"]
        print(f"\nseed={seed}  T={T}  loss={ckpt['loss']:.4f}")
        per_t = {}
        for t in range(1, T + 1):
            agg = aggregate_tc_decomposition(
                model, fp, x, t - 1, args.block_size,
                mc_samples=args.mc_samples,
                n_pair_samples=args.n_pair_samples,
            )
            ratio = agg["within_tc_mean"] / max(
                agg["within_tc_mean"] + agg["between_mi_mean"], 1e-12
            )
            agg["within_fraction"] = ratio
            per_t[t] = agg
            print(f"  t={t}  within_TC={agg['within_tc_mean']:.4f}  "
                  f"between_MI≈{agg['between_mi_mean']:.4f}  "
                  f"within_frac={ratio*100:.1f}%")
        per_seed[seed] = per_t

        del model, fp
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Aggregate over seeds
    print("\n=== aggregate (mean ± sd across seeds) ===")
    ts = sorted(per_seed[next(iter(per_seed))].keys())
    aggregate = {}
    for t in ts:
        w = [per_seed[s][t]["within_tc_mean"] for s in per_seed]
        b = [per_seed[s][t]["between_mi_mean"] for s in per_seed]
        r = [per_seed[s][t]["within_fraction"] for s in per_seed]
        aggregate[t] = {
            "within_tc": {"mean": float(np.mean(w)), "sd": float(np.std(w))},
            "between_mi": {"mean": float(np.mean(b)), "sd": float(np.std(b))},
            "within_fraction": {"mean": float(np.mean(r)), "sd": float(np.std(r))},
            "n_seeds": len(per_seed),
        }
        print(f"  t={t}  within={aggregate[t]['within_tc']['mean']:.4f}"
              f"±{aggregate[t]['within_tc']['sd']:.4f}  "
              f"between≈{aggregate[t]['between_mi']['mean']:.4f}"
              f"±{aggregate[t]['between_mi']['sd']:.4f}  "
              f"within_frac={aggregate[t]['within_fraction']['mean']*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(ts))
    width = 0.35
    within_means = [aggregate[t]["within_tc"]["mean"] for t in ts]
    between_means = [aggregate[t]["between_mi"]["mean"] for t in ts]
    within_sds = [aggregate[t]["within_tc"]["sd"] for t in ts]
    between_sds = [aggregate[t]["between_mi"]["sd"] for t in ts]
    ax.bar(xs - width/2, within_means, width, yerr=within_sds,
           label="within-block TC (absorbed by |G|=4)",
           color="#4393c3", capsize=3, edgecolor="black", linewidth=0.4)
    ax.bar(xs + width/2, between_means, width, yerr=between_sds,
           label="between-block MI≈ (irreducible)",
           color="#d6604d", capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"t={t}" for t in ts])
    ax.set_ylabel("nats per block (or per block-pair for between)")
    ax.set_title("E5: within- vs between-block TC decomposition (|G|=4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.fig_prefix + ".png", dpi=150)
    fig.savefig(args.fig_prefix + ".pdf")
    plt.close(fig)
    print(f"\nwrote {args.fig_prefix}.png/.pdf")

    payload = {
        "config": {
            "ckpt_dir": args.ckpt_dir,
            "block_size": args.block_size,
            "n_images": args.n_images,
            "mc_samples": args.mc_samples,
            "n_pair_samples": args.n_pair_samples,
        },
        "per_seed": {str(s): {str(t): v for t, v in per_seed[s].items()}
                     for s in per_seed},
        "aggregate": {str(t): v for t, v in aggregate.items()},
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.results_json}")


if __name__ == "__main__":
    main()
