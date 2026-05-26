"""E3: block joint analysis on trained |G|=4 MNIST checkpoints.

Loads bs4_s*_best.pt from checkpoints_e2/ and, for a fixed test batch,
computes the within-block TC = KL(joint || product-of-marginals) of the
learned reverse model at each diffusion step t. Aggregates by block
category (background / mixed / stroke), saves a JSON of per-(seed, t,
category) means + counts, and writes two figures:

  - e3_tc_by_category.{png,pdf}: grouped bar chart, mean TC per t per cat
  - e3_block_joint_examples.{png,pdf}: at the t with the largest
    mixed-vs-background gap, three median-TC example blocks per category
    showing joint vs. product-of-marginals as paired bars.

|G|=1 is excluded by construction (within-block TC is identically zero).
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
    classify_blocks,
    factorize_from_marginals,
    joint_to_pixel_marginals,
    within_block_tc,
)
from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet


CATEGORIES = ["background", "mixed", "stroke"]
CATEGORY_COLORS = {
    "background": "#bdbdbd",
    "mixed": "#d6604d",
    "stroke": "#4393c3",
}
CKPT_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    block_size = ckpt["block_size"]
    T = ckpt["T"]
    sigmoid_offset = ckpt.get(
        "sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET
    )
    model = UNet(channels=(32, 64, 128), block_size=block_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    forward_process = LearnedForwardProcess(
        T=T, sigmoid_offset=sigmoid_offset,
    ).to(device)
    forward_process.load_state_dict(ckpt["forward"])
    forward_process.eval()
    return model, forward_process, ckpt


@torch.no_grad()
def joint_at_step(model, forward_process, x, t_idx, block_size):
    """Run the reverse model on z_t ~ q(z_t | x) and return joint probs.

    Returns:
        joint: (B, K^|G|, Hb, Wb)
        z_t:   (B, 1, H, W)
    """
    z_t, _ = forward_process.sample_zt(x, t_idx)
    t_batch = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
    logits = model(z_t, t_batch)
    if block_size == 1:
        p1 = torch.sigmoid(logits)
        joint = torch.cat([1 - p1, p1], dim=1)
    else:
        joint = F.softmax(logits, dim=1)
    return joint, z_t


def get_test_batch(n_images, device, seed=0):
    """Fixed batch of binarized MNIST test images."""
    _, test_loader = get_binarized_mnist(batch_size=n_images)
    g = torch.Generator().manual_seed(seed)
    for (x,) in test_loader:
        # take the first batch — get_binarized_mnist already shuffles deterministically
        return x.to(device)
    raise RuntimeError("empty test loader")


@torch.no_grad()
def aggregate_seed(model, forward_process, T, block_size, x):
    """Per-(t, category) stats for one trained model.

    Returns: dict {t: {cat: {mean, std, count}}}
    """
    categories = classify_blocks(x, block_size).cpu()
    stats = {}
    for t in range(1, T + 1):
        joint, _ = joint_at_step(model, forward_process, x, t - 1, block_size)
        tc = within_block_tc(joint, block_size).cpu()
        per_cat = {}
        for cat_idx, cat_name in enumerate(CATEGORIES):
            mask = categories == cat_idx
            values = tc[mask]
            per_cat[cat_name] = {
                "mean": float(values.mean()) if values.numel() else 0.0,
                "std": float(values.std()) if values.numel() > 1 else 0.0,
                "count": int(values.numel()),
            }
        stats[t] = per_cat
    return stats


def aggregate_across_seeds(per_seed):
    seeds = list(per_seed.keys())
    ts = sorted(per_seed[seeds[0]].keys())
    agg = {}
    for t in ts:
        agg[t] = {}
        for cat in CATEGORIES:
            seed_means = [per_seed[s][t][cat]["mean"] for s in seeds]
            agg[t][cat] = {
                "mean_of_means": float(np.mean(seed_means)),
                "std_of_means": float(np.std(seed_means)),
                "n_seeds": len(seeds),
                "total_blocks": sum(per_seed[s][t][cat]["count"] for s in seeds),
            }
    return agg


def pick_best_t(agg):
    """Pick t with largest mean(mixed) - mean(background)."""
    best_t, best_gap = None, -float("inf")
    for t, by_cat in agg.items():
        gap = by_cat["mixed"]["mean_of_means"] - by_cat["background"]["mean_of_means"]
        if gap > best_gap:
            best_gap = gap
            best_t = t
    return best_t, best_gap


def plot_tc_by_category(agg, out_prefix):
    ts = sorted(agg.keys())
    x = np.arange(len(ts))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cat in enumerate(CATEGORIES):
        means = [agg[t][cat]["mean_of_means"] for t in ts]
        stds = [agg[t][cat]["std_of_means"] for t in ts]
        ax.bar(
            x + (i - 1) * width, means, width, yerr=stds, capsize=3,
            label=cat, color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={t}" for t in ts])
    ax.set_ylabel("within-block TC  KL(joint ‖ product) [nats]")
    ax.set_title(
        "|G|=4: learned block-joint deviation from factorized form\n"
        "(bars = mean across blocks; error bars = std across 3 seeds)"
    )
    ax.legend(title="block category")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=150)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


@torch.no_grad()
def collect_median_examples(model, forward_process, x, t_idx, block_size, n_per_cat=3):
    """For each category, pick blocks at the {25%, 50%, 75%} TC quantiles.

    Returns: dict {cat_name: list of dicts with joint, product, tc, x_patch, zt_patch}
    """
    joint, z_t = joint_at_step(model, forward_process, x, t_idx, block_size)
    marginals = joint_to_pixel_marginals(joint, block_size)
    product = factorize_from_marginals(marginals, block_size)
    tc = within_block_tc(joint, block_size)
    categories = classify_blocks(x, block_size)

    B, _, Hb, Wb = joint.shape
    quantiles = [0.25, 0.5, 0.75][:n_per_cat] if n_per_cat <= 3 else None
    if quantiles is None:
        quantiles = list(np.linspace(0.25, 0.75, n_per_cat))

    out = {}
    for cat_idx, cat_name in enumerate(CATEGORIES):
        mask = categories == cat_idx
        if mask.sum() == 0:
            out[cat_name] = []
            continue
        flat_tc = tc[mask]
        sorted_vals, sorted_idx_in_masked = flat_tc.sort()
        examples = []
        n_masked = flat_tc.numel()
        # map masked-flat-index back to (b, hb, wb)
        b_ids, h_ids, w_ids = mask.nonzero(as_tuple=True)
        for q in quantiles:
            pos = int(q * (n_masked - 1))
            i = int(sorted_idx_in_masked[pos].item())
            b = int(b_ids[i].item())
            h = int(h_ids[i].item())
            w = int(w_ids[i].item())
            examples.append({
                "joint": joint[b, :, h, w].cpu().numpy(),
                "product": product[b, :, h, w].cpu().numpy(),
                "tc": float(tc[b, h, w].cpu()),
                "x_patch": _get_patch(x, b, h, w, block_size),
                "zt_patch": _get_patch(z_t, b, h, w, block_size),
            })
        out[cat_name] = examples
    return out


def _get_patch(img, b, hb, wb, block_size):
    """Extract the underlying pixel patch from a block index (hb, wb)."""
    if block_size == 1:
        return img[b, 0, hb:hb + 1, wb:wb + 1].cpu().numpy()
    if block_size == 2:
        return img[b, 0, hb:hb + 1, 2 * wb:2 * wb + 2].cpu().numpy()
    return img[b, 0, 2 * hb:2 * hb + 2, 2 * wb:2 * wb + 2].cpu().numpy()


def plot_block_joint_examples(examples_by_cat, t_label, block_size, out_prefix):
    n_cats = len(CATEGORIES)
    n_per_cat = max(len(examples_by_cat[c]) for c in CATEGORIES)
    fig, axes = plt.subplots(
        n_cats, n_per_cat, figsize=(3.4 * n_per_cat, 2.6 * n_cats), squeeze=False,
    )
    n_states = 2 ** block_size
    xs = np.arange(n_states)

    for r, cat in enumerate(CATEGORIES):
        for c in range(n_per_cat):
            ax = axes[r, c]
            if c >= len(examples_by_cat[cat]):
                ax.axis("off")
                continue
            ex = examples_by_cat[cat][c]
            ax.bar(xs - 0.2, ex["joint"], width=0.4,
                   label="joint", color=CATEGORY_COLORS[cat],
                   edgecolor="black", linewidth=0.4)
            ax.bar(xs + 0.2, ex["product"], width=0.4,
                   label="product-of-marginals", color="#888888",
                   alpha=0.85, edgecolor="black", linewidth=0.4)
            ax.set_title(f"{cat}  TC = {ex['tc']:.3f}", fontsize=9)
            ax.set_xticks([0, n_states // 2, n_states - 1])
            ax.set_xticklabels(
                [f"{s:0{block_size}b}" for s in
                 (0, n_states // 2, n_states - 1)],
                fontsize=7,
            )
            ax.tick_params(axis="y", labelsize=7)
            if c == 0:
                ax.set_ylabel("probability", fontsize=8)
            if r == n_cats - 1:
                ax.set_xlabel(f"joint state (binary, {block_size} bits)", fontsize=8)

            # inset showing the clean x patch
            inset = ax.inset_axes([0.7, 0.55, 0.27, 0.35])
            inset.imshow(ex["x_patch"], cmap="gray", vmin=0, vmax=1,
                         interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("x", fontsize=7)

            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Block joints vs product-of-marginals at t={t_label}  (|G|={block_size})\n"
        "examples at the 25/50/75% TC quantile within each category",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_prefix + ".png", dpi=150)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_e2")
    parser.add_argument("--block_size", type=int, default=4,
                        help="only |G|>=2 is meaningful (TC is 0 for |G|=1)")
    parser.add_argument("--n_images", type=int, default=2048,
                        help="how many MNIST test images to evaluate over")
    parser.add_argument("--examples_per_cat", type=int, default=3)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_json", type=str, default="results/results_e3.json")
    parser.add_argument("--fig_prefix", type=str, default="figures/e3")
    args = parser.parse_args()

    pattern = os.path.join(args.ckpt_dir, f"bs{args.block_size}_s*_best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {pattern}")
    print(f"E3 block-joint analysis | |G|={args.block_size}")
    print(f"checkpoints ({len(paths)}):")
    for p in paths:
        print(f"  {p}")

    x = get_test_batch(args.n_images, args.device)
    print(f"test batch: {tuple(x.shape)}")

    cat_counts_per_image = classify_blocks(x, args.block_size).flatten().bincount(
        minlength=len(CATEGORIES)).cpu().tolist()
    print(f"block-category counts (bg/mixed/stroke): {cat_counts_per_image}")

    per_seed = {}
    T = None
    first_model = None
    for path in paths:
        m = CKPT_RE.search(os.path.basename(path))
        seed = int(m.group(2)) if m else -1
        model, forward_process, ckpt = load_checkpoint(path, args.device)
        T = ckpt["T"]
        print(f"\nseed={seed}  T={T}  epoch={ckpt['epoch']}  loss={ckpt['loss']:.4f}")
        stats = aggregate_seed(model, forward_process, T, args.block_size, x)
        for t in range(1, T + 1):
            row = stats[t]
            print(
                f"  t={t}  bg={row['background']['mean']:.4f}  "
                f"mixed={row['mixed']['mean']:.4f}  "
                f"stroke={row['stroke']['mean']:.4f}"
            )
        per_seed[seed] = stats
        if first_model is None:
            first_model = (model, forward_process)
        else:
            del model, forward_process
            if args.device == "cuda":
                torch.cuda.empty_cache()

    agg = aggregate_across_seeds(per_seed)
    best_t, best_gap = pick_best_t(agg)
    print(f"\nbest t for examples = {best_t}  (mixed - bg gap = {best_gap:.4f} nats)")

    print("\n=== aggregate (mean across blocks, mean across seeds) ===")
    print(f"{'t':>3} | {'background':>14} | {'mixed':>14} | {'stroke':>14}")
    for t in sorted(agg.keys()):
        print(
            f"{t:>3} | "
            f"{agg[t]['background']['mean_of_means']:>10.4f}"
            f" ± {agg[t]['background']['std_of_means']:.4f} | "
            f"{agg[t]['mixed']['mean_of_means']:>10.4f}"
            f" ± {agg[t]['mixed']['std_of_means']:.4f} | "
            f"{agg[t]['stroke']['mean_of_means']:>10.4f}"
            f" ± {agg[t]['stroke']['std_of_means']:.4f}"
        )

    print("\nrendering figures...")
    plot_tc_by_category(agg, args.fig_prefix + "_tc_by_category")
    print(f"  wrote {args.fig_prefix}_tc_by_category.png/.pdf")

    model, forward_process = first_model
    examples = collect_median_examples(
        model, forward_process, x, best_t - 1, args.block_size,
        n_per_cat=args.examples_per_cat,
    )
    plot_block_joint_examples(
        examples, best_t, args.block_size,
        args.fig_prefix + "_block_joint_examples",
    )
    print(f"  wrote {args.fig_prefix}_block_joint_examples.png/.pdf")

    payload = {
        "config": {
            "ckpt_dir": args.ckpt_dir,
            "block_size": args.block_size,
            "n_images": args.n_images,
            "examples_per_cat": args.examples_per_cat,
            "categories": CATEGORIES,
            "category_counts_in_test_batch": dict(zip(CATEGORIES, cat_counts_per_image)),
            "metric_is_zero_for_block_size_1_by_construction": True,
        },
        "per_seed": {str(s): per_seed[s] for s in per_seed},
        "aggregate": {str(t): agg[t] for t in sorted(agg.keys())},
        "best_t_for_examples": best_t,
        "best_t_mixed_vs_bg_gap": best_gap,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote results -> {args.results_json}")


if __name__ == "__main__":
    main()
