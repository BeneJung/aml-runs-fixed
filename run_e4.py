"""E4: FID vs number of diffusion steps T (stretch goal).

Tests H_E4 — does the block-factorization advantage grow as T decreases?
At low T, each reverse step has more uncertainty about z_{t-1} given z_t
(z_t is more strongly corrupted relative to data), so the data-averaged
target q(z_s | z_t) carries higher TC. A block-factorized reverse head
should absorb relatively more of that local TC, widening Δ FID(|G|=1 - |G|=4)
at smaller T.

Sweeps T in {2, 4, 8, 16} for |G| in {1, 4} over multiple seeds. By default
reuses the existing T=4 best.pt's from checkpoints_e2/ to save compute.

Note: ELBO loss is *not* comparable across T (it scales linearly with T).
Compare ELBO only within a fixed T; compare FID across T values.

Outputs:
  - results_e4.json    per-(T, |G|, seed) FID + ELBO + aggregates
  - e4_fid_vs_t.{png,pdf}    FID-vs-T line chart, one line per |G|
  - e4_gap_vs_t.{png,pdf}    paired Δ FID vs T (block advantage curve)
"""

import argparse
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from train_mnist import run_mnist
from run_e2 import (
    ensure_real_fid_images,
    generate_samples_to_dir,
    compute_fid,
)


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    bs = ckpt["block_size"]
    T = ckpt["T"]
    # B1 backward-compat: ckpts saved by the old code lack sigmoid_offset;
    # default to the historical -2.0 so reproductions match.
    sigmoid_offset = ckpt.get(
        "sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET
    )
    model = UNet(channels=(32, 64, 128), block_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fp = LearnedForwardProcess(T=T, sigmoid_offset=sigmoid_offset).to(device)
    fp.load_state_dict(ckpt["forward"])
    fp.eval()
    return model, fp, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T_values", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_fid_samples", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_e4")
    parser.add_argument("--reuse_t4_dir", type=str, default="checkpoints_e2",
                        help="dir with existing T=4 best.pt's to reuse "
                             "(set to '' to retrain T=4)")
    parser.add_argument("--real_dir", type=str, default="fid_stats/real")
    parser.add_argument("--gen_root", type=str, default="fid_stats_e4")
    parser.add_argument("--keep_gen", action="store_true")
    parser.add_argument("--results_json", type=str, default="results/results_e4.json")
    parser.add_argument("--fig_prefix", type=str, default="figures/e4")
    # forwarded to run_mnist
    parser.add_argument("--sigmoid_offset", type=float,
                        default=LearnedForwardProcess.HISTORICAL_OFFSET,
                        help="Forward-schedule offset (B1).")
    parser.add_argument("--loss_form", type=str, default="ce",
                        choices=["ce", "true_elbo"])
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--select_by", type=str, default="train_loss",
                        choices=["train_loss", "val_loss"])
    parser.add_argument("--val_samples_per_t", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ensure_real_fid_images(args.real_dir)
    print(f"E4 FID-vs-T sweep | T={args.T_values} |G|={args.block_sizes} "
          f"seeds={args.seeds} epochs={args.epochs}")

    results = []
    for T in args.T_values:
        for bs in args.block_sizes:
            for seed in args.seeds:
                key = f"T{T}_bs{bs}_s{seed}"

                reuse_path = None
                if T == 4 and args.reuse_t4_dir:
                    cand = os.path.join(args.reuse_t4_dir,
                                        f"bs{bs}_s{seed}_best.pt")
                    if os.path.exists(cand):
                        reuse_path = cand

                t0 = time.time()
                if reuse_path:
                    print(f"\n=== {key} :: reusing {reuse_path} ===")
                    model, fp, ckpt = load_ckpt(reuse_path, args.device)
                    final_loss = float(ckpt["loss"])
                    best_loss = final_loss
                    best_epoch = int(ckpt["epoch"])
                    epochs_trained = 100  # E2 trained at 100 epochs
                else:
                    print(f"\n=== {key} :: training {args.epochs} epochs ===")
                    r = run_mnist(
                        block_size=bs, seed=seed, T=T, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        device=args.device, save_dir=args.save_dir,
                        save_ckpt_as_best=f"{key}_best.pt",
                        save_ckpt_as_final=f"{key}_final.pt",
                        sample_every=0, samples_dir=None, verbose=True,
                        sigmoid_offset=args.sigmoid_offset,
                        loss_form=args.loss_form,
                        val_fraction=args.val_fraction,
                        select_by=args.select_by,
                        val_samples_per_t=args.val_samples_per_t,
                    )
                    model, fp = r["model"], r["forward_process"]
                    final_loss = float(r["final_loss"])
                    best_loss = float(r["best_loss"])
                    best_epoch = int(r["best_epoch"])
                    epochs_trained = args.epochs
                    print(f"  final_loss={final_loss:.4f} best_loss={best_loss:.4f} "
                          f"best_epoch={best_epoch}")

                gen_dir = os.path.join(args.gen_root, key)
                print(f"  sampling {args.n_fid_samples} -> {gen_dir}")
                generate_samples_to_dir(model, fp, T, bs, args.n_fid_samples,
                                        gen_dir, args.device,
                                        batch_size=args.batch_size)
                print("  computing FID...")
                fid = compute_fid(args.real_dir, gen_dir, args.device)
                elapsed = time.time() - t0
                print(f"  T={T} |G|={bs} seed={seed} FID={fid:.4f} "
                      f"(elapsed {elapsed:.1f}s)")
                if not args.keep_gen:
                    shutil.rmtree(gen_dir)

                results.append({
                    "T": T, "block_size": bs, "seed": seed,
                    "final_loss": final_loss, "best_loss": best_loss,
                    "best_epoch": best_epoch, "epochs_trained": epochs_trained,
                    "fid": float(fid), "reused": reuse_path is not None,
                })
                del model, fp
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    print("\n=== E4 summary ===")
    print(f"{'T':>3} | {'|G|':>3} | {'final_loss (mean)':>18} | "
          f"{'FID (mean ± std)':>22}")
    aggregates = {}
    for T in args.T_values:
        aggregates[T] = {}
        for bs in args.block_sizes:
            rs = [r for r in results if r["T"] == T and r["block_size"] == bs]
            fids = torch.tensor([r["fid"] for r in rs])
            losses = torch.tensor([r["final_loss"] for r in rs])
            fid_std = fids.std(unbiased=False) if len(rs) == 1 else fids.std()
            loss_std = losses.std(unbiased=False) if len(rs) == 1 else losses.std()
            print(f"{T:>3} | {bs:>3} | {losses.mean():>18.4f} | "
                  f"{fids.mean():>10.4f} ± {fid_std:<10.4f}")
            aggregates[T][bs] = {
                "n_seeds": len(rs),
                "fid_mean": float(fids.mean()),
                "fid_std": float(fid_std),
                "loss_mean": float(losses.mean()),
                "loss_std": float(loss_std),
            }

    Ts = args.T_values
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {1: "#4393c3", 4: "#d6604d"}
    for bs in args.block_sizes:
        means = [aggregates[T][bs]["fid_mean"] for T in Ts]
        stds = [aggregates[T][bs]["fid_std"] for T in Ts]
        ax.errorbar(Ts, means, yerr=stds, marker="o", capsize=4,
                    color=colors.get(bs, None), label=f"|G|={bs}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels(Ts)
    ax.set_xlabel("number of diffusion steps T")
    ax.set_ylabel("FID @ 10k samples")
    ax.set_title("E4: FID vs T  (mean ± std across seeds)")
    ax.legend(title="reverse-head factorization")
    fig.tight_layout()
    fig.savefig(args.fig_prefix + "_fid_vs_t.png", dpi=150)
    fig.savefig(args.fig_prefix + "_fid_vs_t.pdf")
    plt.close(fig)
    print(f"\nwrote {args.fig_prefix}_fid_vs_t.png/.pdf")

    if 1 in args.block_sizes and 4 in args.block_sizes:
        gap_means = []
        gap_stds = []
        for T in Ts:
            per_seed = {}
            for r in results:
                if r["T"] != T:
                    continue
                per_seed.setdefault(r["seed"], {})[r["block_size"]] = r["fid"]
            gaps = [per_seed[s][1] - per_seed[s][4]
                    for s in per_seed if 1 in per_seed[s] and 4 in per_seed[s]]
            gap_means.append(float(np.mean(gaps)) if gaps else 0.0)
            gap_stds.append(float(np.std(gaps)) if len(gaps) > 1 else 0.0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(Ts, gap_means, yerr=gap_stds, marker="o", capsize=4,
                    color="#d6604d")
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels(Ts)
        ax.set_xlabel("number of diffusion steps T")
        ax.set_ylabel("Δ FID  =  FID(|G|=1) − FID(|G|=4)")
        ax.set_title("E4: paired block advantage vs T")
        fig.tight_layout()
        fig.savefig(args.fig_prefix + "_gap_vs_t.png", dpi=150)
        fig.savefig(args.fig_prefix + "_gap_vs_t.pdf")
        plt.close(fig)
        print(f"wrote {args.fig_prefix}_gap_vs_t.png/.pdf")

    payload = {
        "config": {
            "T_values": args.T_values, "block_sizes": args.block_sizes,
            "seeds": args.seeds, "epochs": args.epochs,
            "n_fid_samples": args.n_fid_samples,
            "reuse_t4_dir": args.reuse_t4_dir,
        },
        "per_run": results,
        "aggregates": {str(T): {str(bs): aggregates[T][bs]
                                for bs in aggregates[T]}
                       for T in aggregates},
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
