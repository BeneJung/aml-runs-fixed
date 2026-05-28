"""FID Monte-Carlo: score each checkpoint 3 times with different RNG seeds.

Addresses REVIEW.md B9.ii (FID has sample-set noise of its own). For each
matching `bs{}_s{}_<kind>.pt` checkpoint in --ckpt_dir, draws 3 independent
sample sets of `--n_samples` images using seeds {0, 1, 2}, scores FID
against the cached MNIST test set on each, and reports the per-ckpt
mean ± std.

If a checkpoint contains an `"ema"` key, the EMA shadow is used for
sampling (matches what production FID is scored against).

Output JSON has shape:
    {
      "config": {...},
      "per_ckpt": [
        {"path", "block_size", "seed", "kind",
         "fid_seeds": [fid0, fid1, fid2],
         "fid_mean", "fid_std", "used_ema"},
        ...
      ]
    }
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.sample import sample as sample_chain
from fldd.data import get_binarized_mnist
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


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
def generate(model, fp, T, block_size, n_samples, device, generator,
             batch_size=256):
    parts, rem = [], n_samples
    while rem > 0:
        n = min(batch_size, rem)
        parts.append(
            sample_chain(model, fp, T, n_samples=n, device=device,
                         block_size=block_size, generator=generator).cpu()
        )
        rem -= n
    return torch.cat(parts, dim=0)


def fid_once(model, fp, T, block_size, n_samples, device, seed):
    rf = get_real_features(device, n_real=n_samples)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    imgs = generate(model, fp, T, block_size, n_samples, device, gen)
    gf = features(imgs, device)
    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)
    mu_g, sig_g = gf.mean(0), np.cov(gf, rowvar=False)
    return float(calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g))


CKPT_RE = re.compile(r"bs(?P<bs>\d+)_s(?P<seed>\d+)_(?P<kind>best|valbest|final)\.pt$")


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    T = int(ckpt.get("T", 4))
    bs = int(ckpt["block_size"])
    sigmoid_offset = ckpt.get("sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET)
    fixed_alphas = ckpt.get("fixed_alphas", None)
    model = UNet(channels=(32, 64, 128), block_size=bs).to(device)
    model.load_state_dict(ckpt["model"])
    fp = LearnedForwardProcess(
        T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas,
    ).to(device)
    fp.load_state_dict(ckpt["forward"])
    used_ema = False
    if "ema" in ckpt:
        shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name].to(device))
        used_ema = True
    model.eval(); fp.eval()
    return model, fp, T, bs, used_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--ckpt_kind", default="valbest",
                    choices=["best", "valbest", "final"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--fid_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results_json", required=True)
    args = ap.parse_args()

    pattern = os.path.join(args.ckpt_dir, f"bs*_s*_{args.ckpt_kind}.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no checkpoints matching {pattern}")
        return

    print(f"FID MC | {len(paths)} ckpts × {len(args.fid_seeds)} seeds "
          f"× {args.n_samples} samples")
    print(f"  ckpt_kind={args.ckpt_kind} device={args.device}")

    rows = []
    for path in paths:
        m = CKPT_RE.search(os.path.basename(path))
        if not m:
            print(f"  skip (unparseable name): {path}")
            continue
        bs = int(m["bs"]); seed = int(m["seed"])
        print(f"\n--- {os.path.basename(path)} ---")
        t0 = time.time()
        model, fp, T, _bs, used_ema = load_ckpt(path, args.device)
        fids = []
        for s in args.fid_seeds:
            t1 = time.time()
            fid = fid_once(model, fp, T, bs, args.n_samples, args.device, s)
            print(f"  seed={s}  FID={fid:.4f}  ({time.time() - t1:.1f}s)")
            fids.append(fid)
        mean = float(np.mean(fids))
        std = float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0
        print(f"  -> mean {mean:.4f}  std {std:.4f}  ({time.time() - t0:.1f}s total)")
        rows.append({
            "path": path, "block_size": bs, "seed": seed,
            "kind": args.ckpt_kind, "T": T,
            "fid_seeds": fids, "fid_mean": mean, "fid_std": std,
            "used_ema": used_ema,
        })
        del model, fp
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump({
            "config": {
                "ckpt_dir": args.ckpt_dir,
                "ckpt_kind": args.ckpt_kind,
                "n_samples": args.n_samples,
                "fid_seeds": args.fid_seeds,
            },
            "per_ckpt": rows,
        }, f, indent=2)
    print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
