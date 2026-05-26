"""E1: compare pixel- vs block-factorized reverse models on the synthetic dataset.

Trains the configured block_sizes over multiple seeds and prints a summary
table alongside the exact TV floor an optimal pixel-factorized model would hit.

Per-(block_size, seed) we use the same value for model and data seeds, so
each (|G|=1, seed=s) and (|G|=4, seed=s) pair is trained on identical data.
"""

import argparse
import json

import torch

from train_synthetic import run_synthetic
from fldd.synthetic import unconditional_pixel_marginal_tv_floor
from merge_e2_stats import paired_t, wilcoxon, sign_test, bootstrap_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_eval", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_synth")
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--results_json", type=str,
                        default="results/results_e1.json",
                        help="path to dump per-run results + aggregates")
    args = parser.parse_args()

    floor = unconditional_pixel_marginal_tv_floor(epsilon=args.epsilon)
    print(f"device={args.device} T={args.T} epochs={args.epochs} "
          f"epsilon={args.epsilon}")
    print(f"reference: TV to data of best UNCONDITIONAL pixel-marginal "
          f"predictor: {floor:.4f}")
    print("  (NB: this is NOT the floor for pixel-factorized FLDD; the "
          "U-Net conditioning lets it do much better.)\n")

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            print(f"--- training |G|={bs} seed={seed} ---")
            r = run_synthetic(
                block_size=bs, seed=seed, data_seed=seed,
                T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                n_train=args.n_train, n_eval=args.n_eval,
                epsilon=args.epsilon, device=args.device,
                save_dir=args.save_dir,
                save_samples_path=f"{args.samples_dir}/synth_bs{bs}_s{seed}.png",
                verbose=True,
            )
            print(f"    recon={r['final_recon']:.4f}  "
                  f"block-TV={r['block_tv']:.4f}")
            results.append(r)

    print("\n=== E1 summary ===")
    print(f"reference (unconditional pixel-marginal TV): {floor:.4f}  "
          "[NOT a floor for FLDD]")
    print(f"{'|G|':>4} | {'recon (mean±std)':>24} | {'block-TV (mean±std)':>24}")
    aggregates = {}
    for bs in args.block_sizes:
        rs = [r for r in results if r["block_size"] == bs]
        recons = torch.tensor([r["final_recon"] for r in rs])
        tvs = torch.tensor([r["block_tv"] for r in rs])
        recon_std = recons.std(unbiased=False) if len(rs) == 1 else recons.std()
        tv_std = tvs.std(unbiased=False) if len(rs) == 1 else tvs.std()
        print(f"{bs:>4} | "
              f"{recons.mean():>10.4f} ± {recon_std:<10.4f} | "
              f"{tvs.mean():>10.4f} ± {tv_std:<10.4f}")
        aggregates[str(bs)] = {
            "n_seeds": len(rs),
            "recon_mean": float(recons.mean()),
            "recon_std": float(recon_std),
            "block_tv_mean": float(tvs.mean()),
            "block_tv_std": float(tv_std),
        }

    # Paired statistics: each (block_size, seed) shared data, so block-TVs
    # are paired by seed. Reports paired t, Wilcoxon, sign, bootstrap CI on
    # Δ = TV(|G|=1) - TV(|G|=4). H1: block-factorized model has lower TV.
    paired_stats = None
    if 1 in args.block_sizes and 4 in args.block_sizes:
        by_seed = {}
        for r in results:
            by_seed.setdefault(r["seed"], {})[r["block_size"]] = r["block_tv"]
        diffs = []
        for s in sorted(by_seed):
            if 1 in by_seed[s] and 4 in by_seed[s]:
                diffs.append(by_seed[s][1] - by_seed[s][4])
        if diffs:
            t = paired_t(diffs)
            w = wilcoxon(diffs)
            st = sign_test(diffs)
            ci = bootstrap_ci(diffs)
            print("\n=== E1 paired stats (TV(|G|=1) - TV(|G|=4)) ===")
            print(f"  n = {len(diffs)}  mean Δ = {t['mean_diff']:+.4f}  "
                  f"sd Δ = {t['sd_diff']:.4f}")
            print(f"  paired t = {t['t']:.3f}   p(one-sided) = "
                  f"{t['p_one_sided']:.4f}")
            if w:
                print(f"  Wilcoxon p(one-sided) = {w['p_one_sided']:.4f}")
            print(f"  sign test: {st['n_positive']}/{st['n']} favor |G|=4, "
                  f"p(one-sided) = {st['p_one_sided']:.4f}")
            descriptive = len(diffs) < 6
            label = (" [descriptive — n too small for inferential bootstrap]"
                     if descriptive else "")
            print(f"  bootstrap 95% CI on Δ = "
                  f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]{label}")
            paired_stats = {
                "n": len(diffs),
                "diffs": diffs,
                "paired_t": t,
                "wilcoxon": w,
                "sign_test": st,
                "bootstrap_ci": {**ci, "is_descriptive": bool(descriptive)},
            }

    payload = {
        "config": {
            "T": args.T, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "seeds": args.seeds, "block_sizes": args.block_sizes,
            "epsilon": args.epsilon, "n_train": args.n_train,
            "n_eval": args.n_eval,
        },
        "tv_floor_analytic": floor,
        "per_run": [
            {k: (float(v) if isinstance(v, torch.Tensor) else v)
             for k, v in r.items()}
            for r in results
        ],
        "aggregates": aggregates,
        "paired_stats_1v4": paired_stats,
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote results -> {args.results_json}")


if __name__ == "__main__":
    main()
