"""Qualitative sample comparison: ground truth vs pixel baseline vs block models.

Produces a single labeled figure with one panel per column:
    [ Ground truth | |G|=1 (baseline) | |G|=2 | |G|=4 (ours) | ... ]
Each panel is an nrow x nrow grid of binary images. All model panels share
the same starting noise z_T (and RNG), so visible differences are
attributable to the reverse head, not the noise draw.

Ported from the teammate's branch and adapted for our checkpoint naming
(`bs{}_s{}_best.pt` / `_valbest.pt` / `_final.pt`) and EMA support: if a
ckpt contains an `"ema"` key, the EMA shadow weights are used for sampling
(that's what FID is scored against, so the figure should match).

Examples
--------
# MNIST, v2 EMA-on val-best:
python viz_samples.py --dataset mnist --ckpt_dir checkpoints_e2_floor6_ema \\
    --seed 42 --block_sizes 1 2 4 --ckpt_kind valbest \\
    --out figures/e2_samples_comparison.png

# MNIST, original committed v1 ckpts (best.pt selected by train loss):
python viz_samples.py --dataset mnist --ckpt_dir checkpoints_e2 \\
    --seed 42 --block_sizes 1 2 4 --ckpt_kind best \\
    --out figures/e2_samples_v1.png

# Synthetic (E1):
python viz_samples.py --dataset synthetic --ckpt_dir checkpoints_synth \\
    --seed 42 --block_sizes 1 4 --out figures/e1_samples_comparison.png
"""
import argparse
import os

import torch
from torchvision.utils import make_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.sample import sample

UNET_CHANNELS = (32, 64, 128)
LABELS = {1: "|G|=1 (baseline)", 2: "|G|=2", 4: "|G|=4 (ours)"}


def find_ckpt(ckpt_dir, dataset, bs, seed, kind):
    """Resolve a checkpoint path. Tolerates the older 'no-kind' naming
    (`bs{}_s{}.pt` from synthetic runs) and falls back across best/valbest/final."""
    if dataset == "synthetic":
        cands = [f"bs{bs}_s{seed}.pt", f"bs{bs}_s{seed}_best.pt"]
    else:
        cands = [
            f"bs{bs}_s{seed}_{kind}.pt",
            f"bs{bs}_s{seed}_valbest.pt",
            f"bs{bs}_s{seed}_best.pt",
            f"bs{bs}_s{seed}_final.pt",
        ]
    for c in cands:
        p = os.path.join(ckpt_dir, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no checkpoint for bs={bs} seed={seed} in {ckpt_dir} (tried {cands})"
    )


def load_model(path, bs, device, default_T):
    """Load model + forward + EMA (if present). If EMA is present we
    copy the EMA shadow into `model` so subsequent sampling uses it
    (matching what FID would score)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    T = int(ckpt.get("T", default_T))
    block_size = int(ckpt.get("block_size", bs))
    sigmoid_offset = ckpt.get("sigmoid_offset",
                              LearnedForwardProcess.HISTORICAL_OFFSET)
    fixed_alphas = ckpt.get("fixed_alphas", None)
    model = UNet(channels=UNET_CHANNELS, block_size=block_size).to(device)
    model.load_state_dict(ckpt["model"])
    fp = LearnedForwardProcess(
        T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas,
    ).to(device)
    fp.load_state_dict(ckpt["forward"])

    used_ema = False
    if "ema" in ckpt:
        # Copy EMA shadow over the live weights in-place.
        shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name].to(device))
        used_ema = True

    return model, fp, T, ckpt.get("epoch"), used_ema


def ground_truth(dataset, n, device, gen):
    if dataset == "synthetic":
        from fldd.synthetic import sample_synthetic_images, get_ground_truth_block_dist
        imgs = sample_synthetic_images(n, dist=get_ground_truth_block_dist(),
                                       generator=gen)
        return imgs.to(device)
    from fldd.data import get_binarized_mnist
    _, test_loader = get_binarized_mnist(batch_size=n)
    (x,) = next(iter(test_loader))
    return x[:n].to(device)


def panel(ax, imgs, title, nrow):
    grid = make_grid(imgs.cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=0.5)
    ax.imshow(grid[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    ap.add_argument("--ckpt_dir", default="checkpoints_e2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block_sizes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ckpt_kind", default="best",
                    help="best | valbest | final (mnist only)")
    ap.add_argument("--n", type=int, default=64, help="images per panel")
    ap.add_argument("--nrow", type=int, default=8, help="grid columns per panel")
    ap.add_argument("--default_T", type=int, default=4)
    ap.add_argument("--noise_seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="figures/e2_samples_comparison.png")
    args = ap.parse_args()

    dev = args.device
    # Shared starting noise z_T for every model panel (visual differences
    # attributable to the reverse head, not the draw).
    g = torch.Generator(device=dev).manual_seed(args.noise_seed)
    z_init = torch.bernoulli(
        0.5 * torch.ones(args.n, 1, 28, 28, device=dev), generator=g)

    panels = [(
        "Ground truth",
        ground_truth(args.dataset, args.n, dev,
                     torch.Generator(device=dev).manual_seed(args.noise_seed)),
    )]
    for bs in args.block_sizes:
        path = find_ckpt(args.ckpt_dir, args.dataset, bs, args.seed,
                         args.ckpt_kind)
        model, fp, T, epoch, used_ema = load_model(
            path, bs, dev, args.default_T)
        g_s = torch.Generator(device=dev).manual_seed(args.noise_seed + 1)
        imgs = sample(model, fp, T, n_samples=args.n, device=dev,
                      block_size=bs, generator=g_s, z_init=z_init.clone())
        ema_tag = " [EMA]" if used_ema else ""
        tag = LABELS.get(bs, f"|G|={bs}") + ema_tag
        print(f"  {tag}: {os.path.basename(path)} (T={T}, epoch={epoch})")
        panels.append((tag, imgs))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.3))
    if n_panels == 1:
        axes = [axes]
    for ax, (title, imgs) in zip(axes, panels):
        panel(ax, imgs, title, args.nrow)
    fig.suptitle(
        f"{args.dataset.upper()} samples — seed {args.seed} "
        f"({args.ckpt_kind if args.dataset == 'mnist' else 'final'} ckpts, "
        f"shared z_T)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
