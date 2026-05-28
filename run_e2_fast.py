"""Drop-in replacement for run_e2.py that avoids the PNG round-trip.

Same CLI semantics as run_e2.py (same flags), but computes FID in memory
instead of by writing 10k PNGs and reading them back through pytorch_fid.
Solves the rclone bottleneck on Renku where 10k tiny file reads take
hours rather than seconds.

Differences from run_e2.py:
- --gen_root and --real_dir are accepted for compat but ignored
- A single InceptionV3 model is loaded once and reused across all (bs, seed)
- Real MNIST test images are loaded once and reused
- Generated samples never touch disk
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.sample import sample
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from train_mnist import run_mnist


# Cache the InceptionV3 model across runs in the same process
_INC = None
_REAL_FEATS = None


def get_inception(device):
    global _INC
    if _INC is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        _INC = InceptionV3([block_idx]).to(device).eval()
    return _INC


@torch.no_grad()
def features(x_1ch, device, batch_size=50):
    """Per-image: (N, 1, 28, 28) in [0, 1] -> (N, 2048) numpy features."""
    inc = get_inception(device)
    fs = []
    for i in range(0, x_1ch.shape[0], batch_size):
        x = x_1ch[i:i + batch_size].to(device).repeat(1, 3, 1, 1)
        f = inc(x)[0].squeeze(3).squeeze(2)
        fs.append(f.cpu().numpy())
    return np.concatenate(fs, axis=0)


def get_real_features(device, n_real=10000):
    """Compute real-image features once and cache them."""
    global _REAL_FEATS
    if _REAL_FEATS is None:
        _, test_loader = get_binarized_mnist(batch_size=n_real)
        for (x,) in test_loader:
            real = x[:n_real]
            break
        _REAL_FEATS = features(real, device)
        print(f"  cached real features: {_REAL_FEATS.shape}")
    return _REAL_FEATS


@torch.no_grad()
def generate_to_tensor(model, fp, T, block_size, n_samples, device, batch_size=256):
    parts, rem = [], n_samples
    while rem > 0:
        n = min(batch_size, rem)
        parts.append(sample(model, fp, T, n_samples=n,
                            device=device, block_size=block_size).cpu())
        rem -= n
    return torch.cat(parts, dim=0)


def compute_fid_inmem(model, fp, T, block_size, n_samples, device):
    rf = get_real_features(device, n_real=n_samples)
    gens = generate_to_tensor(model, fp, T, block_size, n_samples, device)
    gf = features(gens, device)
    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)
    mu_g, sig_g = gf.mean(0), np.cov(gf, rowvar=False)
    return float(calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_fid_samples", type=int, default=10000)
    p.add_argument("--save_dir", type=str, default="checkpoints_e2_floor6")
    # Backwards-compat (ignored):
    p.add_argument("--real_dir", type=str, default=None)
    p.add_argument("--gen_root", type=str, default=None)
    p.add_argument("--keep_gen", action="store_true")
    p.add_argument("--results_json", type=str, default="results/results_e2.json")
    p.add_argument("--sigmoid_offset", type=float,
                   default=LearnedForwardProcess.HISTORICAL_OFFSET)
    p.add_argument("--loss_form", type=str, default="ce",
                   choices=["ce", "true_elbo"])
    p.add_argument("--val_fraction", type=float, default=0.0)
    p.add_argument("--select_by", type=str, default="train_loss",
                   choices=["train_loss", "val_loss"])
    p.add_argument("--val_samples_per_t", type=int, default=4)
    args = p.parse_args()

    print(f"run_e2_fast | T={args.T} epochs={args.epochs} device={args.device}")
    print(f"  block_sizes={args.block_sizes} seeds={args.seeds} "
          f"n_fid={args.n_fid_samples}")
    print(f"  IN-MEMORY FID (no PNG round-trip)")

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== training |G|={bs} seed={seed} ===")
            r = run_mnist(
                block_size=bs, seed=seed, T=args.T, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                device=args.device, save_dir=args.save_dir,
                save_ckpt_as_best=f"bs{bs}_s{seed}_best.pt",
                save_ckpt_as_final=f"bs{bs}_s{seed}_final.pt",
                sample_every=0, samples_dir=None, verbose=True,
                sigmoid_offset=args.sigmoid_offset,
                loss_form=args.loss_form,
                val_fraction=args.val_fraction,
                select_by=args.select_by,
                val_samples_per_t=args.val_samples_per_t,
            )
            train_secs = time.time() - t0
            print(f"  trained in {train_secs:.1f}s  best_score={r['best_score']:.4f}")

            # Score FID on best.pt (lowest val ELBO), not the final-epoch model.
            # The final-epoch model drifts after the best epoch and gives
            # misleadingly bad FID — we found this the hard way during R1 v2.
            best_path = f"{args.save_dir}/bs{bs}_s{seed}_best.pt"
            print(f"  loading best.pt (epoch {r['best_epoch']}) for FID...")
            best_ckpt = torch.load(best_path, map_location=args.device,
                                    weights_only=False)
            _so = best_ckpt.get("sigmoid_offset", args.sigmoid_offset)
            from fldd.unet import UNet
            best_model = UNet(channels=(32, 64, 128), block_size=bs).to(args.device).eval()
            best_model.load_state_dict(best_ckpt["model"])
            best_fp = LearnedForwardProcess(T=args.T, sigmoid_offset=_so).to(args.device).eval()
            best_fp.load_state_dict(best_ckpt["forward"])

            t1 = time.time()
            print(f"  computing FID (in-memory)...")
            fid = compute_fid_inmem(
                best_model, best_fp,
                args.T, bs, args.n_fid_samples, args.device,
            )
            del best_model, best_fp, best_ckpt
            fid_secs = time.time() - t1
            print(f"  FID = {fid:.4f}  (computed in {fid_secs:.1f}s)")

            del r["model"], r["forward_process"]
            if args.device == "cuda":
                torch.cuda.empty_cache()

            results.append({
                "block_size": bs, "seed": seed,
                "final_loss": r["final_loss"],
                "best_loss": r["best_score"],
                "best_epoch": r["best_epoch"],
                "fid": float(fid),
                "train_secs": train_secs, "fid_secs": fid_secs,
            })

    print("\n=== summary ===")
    for r in results:
        print(f"  |G|={r['block_size']} seed={r['seed']} "
              f"fid={r['fid']:.4f} loss={r['best_loss']:.4f}")

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    payload = {
        "config": {
            "T": args.T, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "seeds": args.seeds, "block_sizes": args.block_sizes,
            "n_fid_samples": args.n_fid_samples,
            "sigmoid_offset": args.sigmoid_offset,
            "loss_form": args.loss_form, "select_by": args.select_by,
            "val_fraction": args.val_fraction,
            "val_samples_per_t": args.val_samples_per_t,
            "fid_mode": "in-memory",
        },
        "per_run": results,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
