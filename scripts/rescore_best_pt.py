"""Rescore FID for any saved best.pt under a checkpoint directory.

Used to recover from the bug where run_e2.py / run_e2_fast.py historically
scored FID against the final-epoch model rather than best.pt. The
rescored numbers are the ones to put in the paper.

Usage:
    cd ~/work/aml-runs-fixed
    python scripts/rescore_best_pt.py \
        --ckpt_dir checkpoints_e2_floor6 \
        --results_dir results/per_run_floor6

Writes one JSON per ckpt named <basename>.json (without _best.pt) into
results_dir, in the format merge_e2_stats.py accepts. Idempotent if the
output JSON already exists (set --force to overwrite).
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.sample import sample
from fldd.unet import UNet
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


CKPT_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")
T2_CKPT_RE = re.compile(r"T(\d+)_bs(\d+)_s(\d+)_frozen_best\.pt$")


def features(x_1ch, inc, device, batch_size=50):
    fs = []
    with torch.no_grad():
        for i in range(0, x_1ch.shape[0], batch_size):
            x = x_1ch[i:i + batch_size].to(device).repeat(1, 3, 1, 1)
            f = inc(x)[0].squeeze(3).squeeze(2)
            fs.append(f.cpu().numpy())
    return np.concatenate(fs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--n_fid_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="overwrite existing JSONs")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*_best.pt")))
    if not paths:
        raise SystemExit(f"no *_best.pt under {args.ckpt_dir}")

    # Cache real features once across all rescorings
    _, test_loader = get_binarized_mnist(batch_size=args.n_fid_samples)
    for (x,) in test_loader:
        real = x[:args.n_fid_samples]
        break

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc = InceptionV3([block_idx]).to(args.device).eval()
    print(f"computing real features ({tuple(real.shape)})...")
    rf = features(real, inc, args.device)
    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)

    os.makedirs(args.results_dir, exist_ok=True)
    summary = []

    for path in paths:
        base = os.path.basename(path).replace("_best.pt", "")
        out_json = os.path.join(args.results_dir, f"{base}.json")
        if os.path.exists(out_json) and not args.force:
            print(f"skip {base} (json exists, pass --force to overwrite)")
            continue

        t0 = time.time()
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        T = ckpt["T"]
        bs = ckpt["block_size"]
        sigmoid_offset = ckpt.get("sigmoid_offset",
                                  LearnedForwardProcess.HISTORICAL_OFFSET)
        fixed_alphas = ckpt.get("fixed_alphas")

        m = UNet(channels=(32, 64, 128), block_size=bs).to(args.device).eval()
        m.load_state_dict(ckpt["model"])
        fp = LearnedForwardProcess(
            T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas
        ).to(args.device).eval()
        fp.load_state_dict(ckpt["forward"])

        # Generate samples and compute features
        gens, rem = [], args.n_fid_samples
        with torch.no_grad():
            while rem > 0:
                n = min(256, rem)
                gens.append(sample(m, fp, T, n_samples=n,
                                   device=args.device, block_size=bs).cpu())
                rem -= n
        gens = torch.cat(gens, dim=0)
        gf = features(gens, inc, args.device)
        mu_g, sig_g = gf.mean(0), np.cov(gf, rowvar=False)
        fid = float(calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g))

        # Parse seed from filename
        m1 = CKPT_RE.search(os.path.basename(path))
        m2 = T2_CKPT_RE.search(os.path.basename(path))
        if m1:
            seed = int(m1.group(2))
        elif m2:
            seed = int(m2.group(3))
        else:
            seed = -1

        payload = {
            "config": {
                "rescored_from_best_pt": True,
                "n_fid_samples": args.n_fid_samples,
                "method": "in-memory FID on best.pt",
                "ckpt_path": path,
            },
            "per_run": [{
                "block_size": int(bs),
                "seed": seed,
                "T": int(T),
                "ckpt_epoch": int(ckpt.get("epoch", -1)),
                "ckpt_loss": float(ckpt.get("val_loss") or ckpt["loss"]),
                "fid": fid,
            }],
            "aggregates": {},
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        secs = time.time() - t0
        print(f"  {base}: FID={fid:.3f}  (epoch {ckpt.get('epoch')}, {secs:.0f}s)")
        summary.append((base, fid, ckpt.get("epoch")))
        del m, fp
        torch.cuda.empty_cache()

    print("\n=== rescored summary ===")
    for base, fid, ep in summary:
        print(f"  {base}: FID={fid:.3f}  (best_epoch={ep})")


if __name__ == "__main__":
    main()
