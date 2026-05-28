"""In-memory FID variant of run_e2.py, v3-audit version.

Differences vs the v2 run_e2_fast.py:
  * Accepts `--ema_decay`. EMA shadow is saved in every checkpoint and
    used for FID scoring (the live weights are *not* used to score FID).
  * Accepts `--save_prefix` (default "bs{bs}_s{seed}"). Useful for the
    pilot-ablation script which writes cells to a single shared dir.
  * Accepts `--cell_name` — purely a label written into the results JSON
    config so `scripts/pick_best_ablation.py` can recover it.
  * Defaults `--loss_form true_elbo` and `--val_samples_per_t 1` to match
    the v3 train_mnist.py defaults.
  * Uses `restore_into_model="auto"` so the model returned by run_mnist()
    is already the val-best (or train-best, depending on select_by) and
    EMA-loaded — we score FID on that, not on a separately-loaded best.pt.
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
    inc = get_inception(device)
    fs = []
    for i in range(0, x_1ch.shape[0], batch_size):
        x = x_1ch[i:i + batch_size].to(device).repeat(1, 3, 1, 1)
        f = inc(x)[0].squeeze(3).squeeze(2)
        fs.append(f.cpu().numpy())
    return np.concatenate(fs, axis=0)


def get_real_features(device, n_real=10000):
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
    p.add_argument("--save_prefix", type=str, default=None,
                   help="If set, ckpts are <save_dir>/<save_prefix>_{best,valbest,final}.pt. "
                        "Default: bs{bs}_s{seed}.")
    p.add_argument("--cell_name", type=str, default=None,
                   help="Label written into results JSON config (for "
                        "scripts/pick_best_ablation.py).")
    p.add_argument("--real_dir", type=str, default=None)  # ignored
    p.add_argument("--gen_root", type=str, default=None)  # ignored
    p.add_argument("--keep_gen", action="store_true")     # ignored
    p.add_argument("--results_json", type=str, default="results/results_e2.json")
    p.add_argument("--sigmoid_offset", type=float,
                   default=LearnedForwardProcess.HISTORICAL_OFFSET)
    p.add_argument("--loss_form", type=str, default="true_elbo",
                   choices=["ce", "true_elbo"])
    p.add_argument("--val_fraction", type=float, default=0.0)
    p.add_argument("--select_by", type=str, default="train_loss",
                   choices=["train_loss", "val_loss"])
    p.add_argument("--val_samples_per_t", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=None,
                   help="EMA decay; e.g. 0.9999. Default off.")
    args = p.parse_args()

    print(f"run_e2_fast (v3) | T={args.T} epochs={args.epochs} device={args.device}")
    print(f"  block_sizes={args.block_sizes} seeds={args.seeds} "
          f"n_fid={args.n_fid_samples}")
    print(f"  sigmoid_offset={args.sigmoid_offset} loss_form={args.loss_form} "
          f"select_by={args.select_by} val_fraction={args.val_fraction} "
          f"ema={args.ema_decay}")
    print(f"  IN-MEMORY FID (no PNG round-trip)")

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== training |G|={bs} seed={seed} ===")

            prefix = args.save_prefix if args.save_prefix else f"bs{bs}_s{seed}"

            r = run_mnist(
                block_size=bs, seed=seed, T=args.T, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                device=args.device, save_dir=args.save_dir,
                save_prefix=prefix,
                sample_every=0, samples_dir=None, verbose=True,
                sigmoid_offset=args.sigmoid_offset,
                loss_form=args.loss_form,
                val_fraction=args.val_fraction,
                val_samples_per_t=args.val_samples_per_t,
                ema_decay=args.ema_decay,
                # run_mnist's "auto" rolls back to valbest if val is on else best
                restore_into_model="auto",
            )
            train_secs = time.time() - t0
            print(f"  trained in {train_secs:.1f}s "
                  f"best_train_loss={r['best_train_loss']:.4f} "
                  f"best_val_loss={r['best_val_loss']}  ")

            # The model returned by run_mnist has the selected ckpt already
            # loaded (EMA copied in if applicable). Score FID directly on it.
            from fldd.train import use_ema
            t1 = time.time()
            print(f"  computing FID (in-memory, on EMA weights={r['ema'] is not None})...")
            with use_ema(r["model"], r["ema"]):
                fid = compute_fid_inmem(
                    r["model"], r["forward_process"],
                    args.T, bs, args.n_fid_samples, args.device,
                )
            fid_secs = time.time() - t1
            print(f"  FID = {fid:.4f}  (computed in {fid_secs:.1f}s)")

            results.append({
                "block_size": bs, "seed": seed,
                "final_loss": r["final_loss"],
                "best_loss": r["best_train_loss"],
                "best_epoch": r["best_train_epoch"],
                "best_val": r["best_val_loss"],
                "best_val_epoch": r["best_val_epoch"],
                "selected_alphas": r["final_alphas"],
                "fid": float(fid),
                "train_secs": train_secs, "fid_secs": fid_secs,
                "restored_kind": (r["restored"] or {}).get("which"),
                "restored_epoch": (r["restored"] or {}).get("epoch"),
                "ema_decay": args.ema_decay,
            })

            del r["model"], r["forward_process"]
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print("\n=== summary ===")
    for r in results:
        print(f"  |G|={r['block_size']} seed={r['seed']} "
              f"fid={r['fid']:.4f} best_loss={r['best_loss']:.4f}")

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
            "ema_decay": args.ema_decay,
            "cell_name": args.cell_name,
            "fid_mode": "in-memory",
        },
        "per_run": results,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
