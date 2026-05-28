"""R3 fast variant: T=2 frozen schedule, in-memory FID, best.pt scoring.

Same scientific goal as run_e4_t2_frozen.py — train two block sizes on a
COMMON frozen schedule to remove the v1 T=2 schedule confound (REVIEW.md
B4) — but uses run_e2_fast.py's in-memory InceptionV3 features so FID
takes ~80s instead of hours on rclone-mounted storage. Also scores FID
on best.pt rather than the final-epoch model.
"""

import argparse
import json
import os
import time

import torch

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from train_mnist import run_mnist
from run_e2_fast import compute_fid_inmem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_fid_samples", type=int, default=10000)
    p.add_argument("--alphas", type=float, nargs="+", required=True,
                   help="Frozen schedule alphas, length must equal T.")
    p.add_argument("--save_dir", type=str, default="checkpoints_e4_t2_frozen")
    p.add_argument("--loss_form", type=str, default="true_elbo",
                   choices=["ce", "true_elbo"])
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--select_by", type=str, default="val_loss",
                   choices=["train_loss", "val_loss"])
    p.add_argument("--val_samples_per_t", type=int, default=2)
    p.add_argument("--results_json", type=str, required=True)
    args = p.parse_args()

    if len(args.alphas) != args.T:
        raise SystemExit(
            f"--alphas needs {args.T} values, got {len(args.alphas)}"
        )
    print(f"R3 fast | T={args.T} alphas={args.alphas} "
          f"block_sizes={args.block_sizes} seeds={args.seeds}")

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== training T={args.T} |G|={bs} seed={seed} ===")
            r = run_mnist(
                block_size=bs, seed=seed, T=args.T, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                device=args.device, save_dir=args.save_dir,
                save_ckpt_as_best=f"T{args.T}_bs{bs}_s{seed}_frozen_best.pt",
                save_ckpt_as_final=f"T{args.T}_bs{bs}_s{seed}_frozen_final.pt",
                sample_every=0, samples_dir=None, verbose=True,
                fixed_alphas=args.alphas,
                loss_form=args.loss_form,
                val_fraction=args.val_fraction,
                select_by=args.select_by,
                val_samples_per_t=args.val_samples_per_t,
            )
            train_secs = time.time() - t0
            print(f"  trained in {train_secs:.1f}s  "
                  f"best_score={r['best_score']:.4f} "
                  f"best_epoch={r['best_epoch']}")

            # Score FID on best.pt (not final-epoch model)
            best_path = (f"{args.save_dir}/T{args.T}_bs{bs}_s{seed}"
                         f"_frozen_best.pt")
            print(f"  loading best.pt for FID...")
            best_ckpt = torch.load(best_path, map_location=args.device,
                                    weights_only=False)
            best_model = UNet(channels=(32, 64, 128), block_size=bs).to(
                args.device).eval()
            best_model.load_state_dict(best_ckpt["model"])
            best_fp = LearnedForwardProcess(
                T=args.T, fixed_alphas=args.alphas).to(args.device).eval()
            best_fp.load_state_dict(best_ckpt["forward"])

            t1 = time.time()
            print(f"  computing FID (in-memory)...")
            fid = compute_fid_inmem(
                best_model, best_fp,
                args.T, bs, args.n_fid_samples, args.device,
            )
            fid_secs = time.time() - t1
            print(f"  T={args.T} bs={bs} s={seed}  FID={fid:.4f}  "
                  f"(train {train_secs:.0f}s, fid {fid_secs:.0f}s)")

            results.append({
                "T": args.T,
                "block_size": bs,
                "seed": seed,
                "fid": float(fid),
                "best_score": r["best_score"],
                "best_epoch": r["best_epoch"],
                "frozen_alphas": list(map(float, args.alphas)),
                "train_secs": train_secs,
                "fid_secs": fid_secs,
            })

            del best_model, best_fp, best_ckpt
            del r["model"], r["forward_process"]
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print("\n=== summary ===")
    for r in results:
        print(f"  T={r['T']} bs={r['block_size']} s={r['seed']}  "
              f"FID={r['fid']:.4f}")

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    payload = {
        "config": {
            "T": args.T,
            "block_sizes": args.block_sizes,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "alphas": list(args.alphas),
            "loss_form": args.loss_form,
            "select_by": args.select_by,
            "val_fraction": args.val_fraction,
            "val_samples_per_t": args.val_samples_per_t,
            "n_fid_samples": args.n_fid_samples,
            "fid_mode": "in-memory",
            "scored_on": "best.pt",
        },
        "per_run": results,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
