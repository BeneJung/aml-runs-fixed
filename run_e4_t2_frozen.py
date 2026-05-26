"""E4 follow-up: T=2 FID with the forward schedule FROZEN across block sizes.

Addresses the B4 confound in the original E4: at T=2 the learned schedules
diverge between |G|=1 (alpha_T ~= 0.38) and |G|=4 (alpha_T ~= 0.50), which
mixes the reverse-head effect with a forward-schedule effect. Re-training
both block sizes on a common, frozen schedule isolates the reverse-head
contribution.

Three ways to pick the frozen schedule:

  --schedule from_bs4_ckpt   Use |G|=4's learned schedule from a saved ckpt.
                             This is the most generous-to-baseline choice
                             (uses the schedule the better model 'wanted').
  --schedule explicit        Pass --alphas a1 a2 ... aT directly.
  --schedule uniform         Set every alpha to 0.5 / T * t (linearly
                             ramping toward 0.5). A neutral default.

Example
-------

  # 1. From the existing |G|=4 T=2 ckpts (default expects checkpoints_e4/T2_bs4_s*_best.pt)
  python run_e4_t2_frozen.py --schedule from_bs4_ckpt \
      --seeds 42 43 44 --device cuda

  # 2. Explicit alphas
  python run_e4_t2_frozen.py --schedule explicit --alphas 0.06 0.50 \
      --seeds 42 43 44 --device cuda

Outputs results/results_e4_t2_frozen.json with per-(bs, seed) FID, and
prints a paired summary at the end.
"""

import argparse
import glob
import json
import os
import re
import shutil
import time

import torch

from fldd.forward import LearnedForwardProcess
from train_mnist import run_mnist
from run_e2 import (
    ensure_real_fid_images,
    generate_samples_to_dir,
    compute_fid,
)


CKPT_RE = re.compile(r"T(\d+)_bs(\d+)_s(\d+)_best\.pt$")


def load_alphas_from_ckpt(ckpt_path, T):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigmoid_offset = ckpt.get("sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET)
    fp = LearnedForwardProcess(T=T, sigmoid_offset=sigmoid_offset)
    fp.load_state_dict(ckpt["forward"])
    with torch.no_grad():
        return fp.get_alphas().tolist()


def pick_bs4_alphas(t2_bs4_dir, T, seed):
    """Prefer a seed-matched bs4 ckpt; fall back to first available."""
    target = os.path.join(t2_bs4_dir, f"T{T}_bs4_s{seed}_best.pt")
    if os.path.exists(target):
        return load_alphas_from_ckpt(target, T), target
    candidates = sorted(glob.glob(os.path.join(t2_bs4_dir, f"T{T}_bs4_s*_best.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"no T={T} bs4 checkpoints found in {t2_bs4_dir}"
        )
    return load_alphas_from_ckpt(candidates[0], T), candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_fid_samples", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_e4_t2_frozen")
    parser.add_argument("--real_dir", type=str, default="fid_stats/real")
    parser.add_argument("--gen_root", type=str, default="fid_stats_e4_t2_frozen")
    parser.add_argument("--results_json", type=str,
                        default="results/results_e4_t2_frozen.json")
    parser.add_argument("--schedule", type=str, default="from_bs4_ckpt",
                        choices=["from_bs4_ckpt", "explicit", "uniform"])
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="for --schedule explicit")
    parser.add_argument("--bs4_ckpt_dir", type=str, default="checkpoints_e4")
    # Pass through to run_mnist; allow opting into the new fixes here.
    parser.add_argument("--loss_form", type=str, default="ce",
                        choices=["ce", "true_elbo"])
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--select_by", type=str, default="train_loss",
                        choices=["train_loss", "val_loss"])
    parser.add_argument("--keep_gen", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ensure_real_fid_images(args.real_dir)

    # --- pick the frozen schedule ---
    if args.schedule == "explicit":
        if args.alphas is None or len(args.alphas) != args.T:
            raise SystemExit(
                f"--schedule explicit requires --alphas with {args.T} values"
            )
        chosen_alphas = list(args.alphas)
        schedule_src = "explicit"
    elif args.schedule == "uniform":
        # ramp linearly to 0.5 (avoids the parameterization saturation)
        chosen_alphas = [0.5 * (t + 1) / args.T for t in range(args.T)]
        # nudge below 0.5 to satisfy the strict (0, 0.5) constraint
        chosen_alphas = [min(a, 0.499) for a in chosen_alphas]
        schedule_src = "uniform"

    print(f"T={args.T}  block_sizes={args.block_sizes}  seeds={args.seeds}")
    print(f"  loss_form={args.loss_form}  select_by={args.select_by}")

    results = []
    for seed in args.seeds:
        if args.schedule == "from_bs4_ckpt":
            chosen_alphas, src_path = pick_bs4_alphas(
                args.bs4_ckpt_dir, args.T, seed
            )
            schedule_src = src_path
            print(f"\n[seed={seed}] frozen alphas = "
                  f"{[round(a, 4) for a in chosen_alphas]}  (from {src_path})")
        else:
            print(f"\n[seed={seed}] frozen alphas = "
                  f"{[round(a, 4) for a in chosen_alphas]}  (source: {schedule_src})")

        for bs in args.block_sizes:
            key = f"T{args.T}_bs{bs}_s{seed}_frozen"
            print(f"\n=== {key} ===")

            t0 = time.time()
            r = run_mnist(
                block_size=bs, seed=seed, T=args.T, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                device=args.device, save_dir=args.save_dir,
                save_ckpt_as_best=f"{key}_best.pt",
                save_ckpt_as_final=f"{key}_final.pt",
                sample_every=0, samples_dir=None, verbose=True,
                fixed_alphas=chosen_alphas,
                loss_form=args.loss_form,
                val_fraction=args.val_fraction,
                select_by=args.select_by,
            )
            print(f"  final_loss={r['final_loss']:.4f}  "
                  f"best_score={r['best_score']:.4f}  "
                  f"best_epoch={r['best_epoch']}")

            gen_dir = os.path.join(args.gen_root, key)
            print(f"  sampling {args.n_fid_samples} -> {gen_dir}")
            generate_samples_to_dir(
                r["model"], r["forward_process"], args.T, bs,
                args.n_fid_samples, gen_dir, args.device,
            )
            fid = compute_fid(args.real_dir, gen_dir, args.device)
            elapsed = time.time() - t0
            print(f"  T={args.T} bs={bs} seed={seed} FID={fid:.4f}  "
                  f"({elapsed:.1f}s)")

            if not args.keep_gen:
                shutil.rmtree(gen_dir)

            results.append({
                "T": args.T, "block_size": bs, "seed": seed,
                "fid": float(fid),
                "final_loss": float(r["final_loss"]),
                "best_score": float(r["best_score"]),
                "best_epoch": int(r["best_epoch"]) if r["best_epoch"] else None,
                "frozen_alphas": list(map(float, chosen_alphas)),
                "schedule_source": schedule_src,
                "loss_form": args.loss_form,
                "select_by": args.select_by,
            })

            del r
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # paired summary
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], {})[r["block_size"]] = r["fid"]
    diffs = []
    print("\n=== paired summary (FID(|G|=1) - FID(|G|=4)) ===")
    for s in sorted(by_seed):
        if 1 in by_seed[s] and 4 in by_seed[s]:
            d = by_seed[s][1] - by_seed[s][4]
            diffs.append(d)
            print(f"  seed {s}: bs1={by_seed[s][1]:.4f}  bs4={by_seed[s][4]:.4f}  "
                  f"Δ={d:+.4f}")
    if diffs:
        mean = sum(diffs) / len(diffs)
        print(f"  mean Δ = {mean:+.4f}  (n={len(diffs)})")

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    payload = {
        "config": {
            "T": args.T, "block_sizes": args.block_sizes, "seeds": args.seeds,
            "epochs": args.epochs, "n_fid_samples": args.n_fid_samples,
            "schedule": args.schedule,
            "loss_form": args.loss_form, "select_by": args.select_by,
            "val_fraction": args.val_fraction,
        },
        "per_run": results,
        "paired_diffs": diffs,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
