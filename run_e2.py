"""E2: MNIST FID sweep comparing pixel- vs block-factorized reverse models.

For each (block_size, seed): trains FLDD on binarized MNIST, generates
n_fid_samples samples, computes FID against cached real test images via
pytorch_fid, and prints a summary table.

Generated image dirs are deleted after FID to save disk (pass --keep_gen to
preserve). Real FID images are cached once on first run.
"""

import argparse
import json
import os
import shutil
import torch
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.sample import sample
from train_mnist import run_mnist


def save_images_for_fid(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, os.path.join(out_dir, f"{i:05d}.png"))


def ensure_real_fid_images(real_dir, batch_size=256):
    if os.path.exists(real_dir) and len(os.listdir(real_dir)) > 0:
        return
    os.makedirs(real_dir, exist_ok=True)
    print(f"caching real MNIST test images -> {real_dir}")
    _, test_loader = get_binarized_mnist(batch_size=batch_size)
    imgs = []
    for (x,) in test_loader:
        imgs.append(x.repeat(1, 3, 1, 1))
    imgs = torch.cat(imgs, dim=0)
    save_images_for_fid(imgs, real_dir)
    print(f"  cached {len(imgs)} real images")


def generate_samples_to_dir(model, forward_process, T, block_size, n_samples,
                            out_dir, device, batch_size=256):
    os.makedirs(out_dir, exist_ok=True)
    all_samples = []
    remaining = n_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        s = sample(model, forward_process, T, n_samples=n,
                   device=device, block_size=block_size)
        all_samples.append(s.repeat(1, 3, 1, 1).cpu())
        remaining -= n
    all_samples = torch.cat(all_samples, dim=0)
    save_images_for_fid(all_samples, out_dir)


def compute_fid(real_dir, gen_dir, device):
    return calculate_fid_given_paths(
        [real_dir, gen_dir], batch_size=50, device=device,
        dims=2048, num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_fid_samples", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_e2")
    parser.add_argument("--real_dir", type=str, default="fid_stats/real")
    parser.add_argument("--gen_root", type=str, default="fid_stats_e2")
    parser.add_argument("--keep_gen", action="store_true",
                        help="keep generated image dirs after FID")
    parser.add_argument("--results_json", type=str, default="results/results_e2.json",
                        help="path to dump per-run results + aggregates")
    # forwarded to train_mnist.run_mnist; default to historical behaviour
    parser.add_argument("--sigmoid_offset", type=float,
                        default=LearnedForwardProcess.HISTORICAL_OFFSET,
                        help="Forward-schedule parameterization offset (B1). "
                             "Use -6.0 to give the schedule a much lower floor "
                             "and verify whether it actually learns away from "
                             "the floor.")
    parser.add_argument("--loss_form", type=str, default="ce",
                        choices=["ce", "true_elbo"],
                        help="'ce' (historical) or 'true_elbo' (strict KL, B5).")
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Held-out validation fraction (B6).")
    parser.add_argument("--select_by", type=str, default="train_loss",
                        choices=["train_loss", "val_loss"],
                        help="Best-ckpt selection criterion (B6).")
    args = parser.parse_args()

    print(f"E2 MNIST FID sweep | T={args.T} epochs={args.epochs} "
          f"device={args.device}")
    print(f"block_sizes={args.block_sizes} seeds={args.seeds} "
          f"n_fid_samples={args.n_fid_samples}")

    ensure_real_fid_images(args.real_dir)

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            print(f"\n=== training |G|={bs} seed={seed} ===")
            r = run_mnist(
                block_size=bs, seed=seed, T=args.T, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                device=args.device, save_dir=args.save_dir,
                save_ckpt_as_best=f"bs{bs}_s{seed}_best.pt",
                save_ckpt_as_final=f"bs{bs}_s{seed}_final.pt",
                sample_every=0, samples_dir=None,
                verbose=True,
                sigmoid_offset=args.sigmoid_offset,
                loss_form=args.loss_form,
                val_fraction=args.val_fraction,
                select_by=args.select_by,
            )
            print(f"  final_loss={r['final_loss']:.4f} "
                  f"best_loss={r['best_loss']:.4f} "
                  f"best_epoch={r['best_epoch']}")
            print(f"  alphas: {[round(a, 3) for a in r['final_alphas']]}")

            gen_dir = os.path.join(args.gen_root, f"bs{bs}_s{seed}")
            print(f"  generating {args.n_fid_samples} samples -> {gen_dir}")
            generate_samples_to_dir(
                r["model"], r["forward_process"], args.T, bs,
                args.n_fid_samples, gen_dir, args.device,
            )

            print(f"  computing FID...")
            fid = compute_fid(args.real_dir, gen_dir, args.device)
            print(f"  FID={fid:.4f}")

            if not args.keep_gen:
                shutil.rmtree(gen_dir)

            del r["model"], r["forward_process"]
            if args.device == "cuda":
                torch.cuda.empty_cache()

            results.append({
                "block_size": bs, "seed": seed,
                "final_loss": r["final_loss"],
                "best_loss": r["best_loss"],
                "best_epoch": r["best_epoch"],
                "fid": fid,
            })

    print("\n=== E2 summary ===")
    print(f"{'|G|':>4} | {'final_loss (mean)':>18} | "
          f"{'best_loss (mean)':>18} | {'FID (mean ± std)':>22}")
    aggregates = {}
    for bs in args.block_sizes:
        rs = [r for r in results if r["block_size"] == bs]
        fids = torch.tensor([r["fid"] for r in rs])
        final_loss = torch.tensor([r["final_loss"] for r in rs])
        best_loss = torch.tensor([r["best_loss"] for r in rs])
        fid_std = fids.std(unbiased=False) if len(rs) == 1 else fids.std()
        print(f"{bs:>4} | {final_loss.mean():>18.4f} | "
              f"{best_loss.mean():>18.4f} | "
              f"{fids.mean():>10.4f} ± {fid_std:<10.4f}")
        aggregates[bs] = {
            "n_seeds": len(rs),
            "final_loss_mean": float(final_loss.mean()),
            "final_loss_std": float(final_loss.std(unbiased=False)
                                    if len(rs) == 1 else final_loss.std()),
            "best_loss_mean": float(best_loss.mean()),
            "best_loss_std": float(best_loss.std(unbiased=False)
                                   if len(rs) == 1 else best_loss.std()),
            "fid_mean": float(fids.mean()),
            "fid_std": float(fid_std),
        }
    print("\nper-run:")
    for r in results:
        print(f"  |G|={r['block_size']} seed={r['seed']} "
              f"fid={r['fid']:.4f} final_loss={r['final_loss']:.4f}")

    payload = {
        "config": {
            "T": args.T, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "seeds": args.seeds, "block_sizes": args.block_sizes,
            "n_fid_samples": args.n_fid_samples,
        },
        "per_run": results,
        "aggregates": {str(k): v for k, v in aggregates.items()},
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote results -> {args.results_json}")


if __name__ == "__main__":
    main()
