"""Re-score existing E2 best.pt checkpoints: sample, compute FID, dump JSON."""

import argparse
import glob
import json
import os
import re
import shutil
import time
import torch

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from run_e2 import (
    ensure_real_fid_images,
    generate_samples_to_dir,
    compute_fid,
)


CKPT_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_e2")
    parser.add_argument("--n_fid_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--real_dir", type=str, default="fid_stats/real")
    parser.add_argument("--gen_root", type=str, default="fid_stats_e2")
    parser.add_argument("--keep_gen", action="store_true")
    parser.add_argument("--results_json", type=str,
                        default="results/results_e2_from_ckpts.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*_best.pt")))
    runs = []
    for p in paths:
        m = CKPT_RE.search(os.path.basename(p))
        if not m:
            continue
        runs.append((int(m.group(1)), int(m.group(2)), p))
    print(f"found {len(runs)} checkpoints in {args.ckpt_dir}")

    ensure_real_fid_images(args.real_dir)

    results = []
    for bs, seed, path in runs:
        t0 = time.time()
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        T = ckpt["T"]
        block_size = ckpt["block_size"]
        sigmoid_offset = ckpt.get(
            "sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET,
        )
        assert block_size == bs

        model = UNet(channels=(32, 64, 128), block_size=block_size).to(args.device)
        model.load_state_dict(ckpt["model"])
        forward_process = LearnedForwardProcess(
            T=T, sigmoid_offset=sigmoid_offset,
        ).to(args.device)
        forward_process.load_state_dict(ckpt["forward"])

        gen_dir = os.path.join(args.gen_root, f"bs{bs}_s{seed}")
        print(f"\n=== |G|={bs} seed={seed} (epoch {ckpt['epoch']}, "
              f"loss {ckpt['loss']:.4f}) ===")
        print(f"  sampling {args.n_fid_samples} -> {gen_dir}")
        t_sample = time.time()
        generate_samples_to_dir(model, forward_process, T, bs,
                                args.n_fid_samples, gen_dir, args.device,
                                batch_size=args.batch_size)
        t_sample = time.time() - t_sample

        print(f"  computing FID...")
        t_fid = time.time()
        fid = compute_fid(args.real_dir, gen_dir, args.device)
        t_fid = time.time() - t_fid

        elapsed = time.time() - t0
        print(f"  FID={fid:.4f}  (sample={t_sample:.1f}s fid={t_fid:.1f}s "
              f"total={elapsed:.1f}s)")

        if not args.keep_gen:
            shutil.rmtree(gen_dir)

        results.append({
            "block_size": bs, "seed": seed,
            "ckpt_epoch": ckpt["epoch"],
            "ckpt_loss": float(ckpt["loss"]),
            "fid": float(fid),
            "sample_seconds": t_sample,
            "fid_seconds": t_fid,
        })

        del model, forward_process, ckpt
        if args.device == "cuda":
            torch.cuda.empty_cache()

    block_sizes = sorted({r["block_size"] for r in results})
    print("\n=== summary ===")
    print(f"{'|G|':>4} | {'ckpt_loss (mean)':>18} | {'FID (mean ± std)':>22}")
    aggregates = {}
    for bs in block_sizes:
        rs = [r for r in results if r["block_size"] == bs]
        fids = torch.tensor([r["fid"] for r in rs])
        losses = torch.tensor([r["ckpt_loss"] for r in rs])
        fid_std = fids.std(unbiased=False) if len(rs) == 1 else fids.std()
        loss_std = losses.std(unbiased=False) if len(rs) == 1 else losses.std()
        print(f"{bs:>4} | {losses.mean():>18.4f} | "
              f"{fids.mean():>10.4f} ± {fid_std:<10.4f}")
        aggregates[bs] = {
            "n_seeds": len(rs),
            "ckpt_loss_mean": float(losses.mean()),
            "ckpt_loss_std": float(loss_std),
            "fid_mean": float(fids.mean()),
            "fid_std": float(fid_std),
        }

    payload = {
        "config": {
            "ckpt_dir": args.ckpt_dir,
            "n_fid_samples": args.n_fid_samples,
            "real_dir": args.real_dir,
        },
        "per_run": results,
        "aggregates": {str(k): v for k, v in aggregates.items()},
    }
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote results -> {args.results_json}")


if __name__ == "__main__":
    main()
