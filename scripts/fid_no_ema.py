"""D1 — the decisive cheap diagnostic for the v3 FID reversal.

Re-scores each v3 checkpoint TWICE on the SAME training trajectory:

  * "live"  — the live U-Net weights  (ckpt["model"]).  Because EMA never
              feeds back into training (fldd/train.py: ema.update() runs after
              optimizer.step() and writes only its own shadow), these live
              weights are exactly what a NO-EMA run with the same seed would
              have produced at this epoch.
  * "ema"   — the EMA shadow weights  (ckpt["ema"]), swapped in via the same
              `use_ema` context manager run_e2_fast.py used to score v3. This
              reproduces the v3 chain's FID numbers.

It then prints, per block size, FID(live) and FID(ema), and the paired
Δ FID(1−4) under each weighting. Interpretation:

  * Δ FID(1−4) on LIVE weights > 0  → block head wins without the EMA shadow
                                       → favors interpretation (b) (EMA artifact)
  * Δ FID(1−4) on LIVE weights < 0  → block still loses on live weights
                                       → favors interpretation (a) (real ELBO–FID gap)

This is read-only: no training, ~15–25 min on a single GPU for 6 ckpts × 2.

Usage (Renku):
    cd ~/work/aml-runs-fixed
    python scripts/fid_no_ema.py \
        --ckpt_dir checkpoints_v3 --ckpt_glob '*_valbest.pt' \
        --n_fid_samples 10000 \
        --out results/v3/fid_no_ema.json

Reconstruction (UNet/LearnedForwardProcess args, real-feature caching, the
in-memory FID) is copied verbatim from scripts/rescore_best_pt.py so this
scores identically to the rest of the pipeline.
"""

import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch

# Make `fldd` importable when invoked as `python scripts/fid_no_ema.py` from
# the repo root: Python puts scripts/ on sys.path[0], not the repo root, so
# prepend the repo root (the parent of scripts/) explicitly.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.sample import sample
from fldd.train import EMA, use_ema
from fldd.unet import UNet
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


# Accept valbest / best / final, with or without a Tn_ prefix (E4-style).
CKPT_RE = re.compile(r"(?:T(\d+)_)?bs(\d+)_s(\d+)_(?:valbest|best|final)\.pt$")


def features(x_1ch, inc, device, batch_size=50):
    fs = []
    with torch.no_grad():
        for i in range(0, x_1ch.shape[0], batch_size):
            x = x_1ch[i:i + batch_size].to(device).repeat(1, 3, 1, 1)
            f = inc(x)[0].squeeze(3).squeeze(2)
            fs.append(f.cpu().numpy())
    return np.concatenate(fs, axis=0)


@torch.no_grad()
def fid_for_model(model, fp, T, bs, n_samples, inc, mu_r, sig_r, device):
    gens, rem = [], n_samples
    while rem > 0:
        n = min(256, rem)
        gens.append(sample(model, fp, T, n_samples=n,
                           device=device, block_size=bs).cpu())
        rem -= n
    gens = torch.cat(gens, dim=0)
    gf = features(gens, inc, device)
    mu_g, sig_g = gf.mean(0), np.cov(gf, rowvar=False)
    return float(calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g))


def paired_delta(rows, key, bs_a=1, bs_b=4):
    """Mean paired Δ = FID(bs_a) − FID(bs_b), paired by seed, on rows[*][key]."""
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["block_size"]] = r[key]
    diffs = []
    for seed, d in sorted(by_seed.items()):
        if bs_a in d and bs_b in d:
            diffs.append((seed, d[bs_a] - d[bs_b]))
    if not diffs:
        return None
    vals = [v for _, v in diffs]
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    out = {"n": n, "mean_diff": mean, "sd_diff": sd,
           "per_seed": {str(s): v for s, v in diffs}}
    if n > 1 and sd > 0:
        t = mean / (sd / (n ** 0.5))
        out["t"] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_glob", default="*_valbest.pt",
                   help="glob within ckpt_dir; v3 used select_by=val_loss so "
                        "the selected ckpt is *_valbest.pt.")
    p.add_argument("--n_fid_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/v3/fid_no_ema.json")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)))
    if not paths:
        raise SystemExit(f"no {args.ckpt_glob} under {args.ckpt_dir}")
    print(f"found {len(paths)} ckpts under {args.ckpt_dir}/{args.ckpt_glob}")

    # Cache real features once.
    _, test_loader = get_binarized_mnist(batch_size=args.n_fid_samples)
    for (x,) in test_loader:
        real = x[:args.n_fid_samples]
        break
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc = InceptionV3([block_idx]).to(args.device).eval()
    print(f"computing real features {tuple(real.shape)}...")
    rf = features(real, inc, args.device)
    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)

    rows = []
    for path in paths:
        m = CKPT_RE.search(os.path.basename(path))
        seed = int(m.group(3)) if m else -1

        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        T = ckpt["T"]
        bs = ckpt["block_size"]
        sigmoid_offset = ckpt.get("sigmoid_offset",
                                  LearnedForwardProcess.HISTORICAL_OFFSET)
        fixed_alphas = ckpt.get("fixed_alphas")
        has_ema = "ema" in ckpt and ckpt["ema"] is not None

        model = UNet(channels=(32, 64, 128), block_size=bs).to(args.device).eval()
        model.load_state_dict(ckpt["model"])
        fp = LearnedForwardProcess(
            T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas
        ).to(args.device).eval()
        fp.load_state_dict(ckpt["forward"])

        t0 = time.time()
        fid_live = fid_for_model(model, fp, T, bs, args.n_fid_samples,
                                 inc, mu_r, sig_r, args.device)

        fid_ema = None
        if has_ema:
            ema = EMA(model, decay=float(ckpt["ema"].get("decay", 0.9999)))
            ema.load_state_dict(ckpt["ema"], device=args.device)
            with use_ema(model, ema):
                fid_ema = fid_for_model(model, fp, T, bs, args.n_fid_samples,
                                        inc, mu_r, sig_r, args.device)

        secs = time.time() - t0
        row = {"block_size": int(bs), "seed": seed, "T": int(T),
               "ckpt_epoch": int(ckpt.get("epoch", -1)),
               "fid_live": fid_live, "fid_ema": fid_ema,
               "ckpt_path": path}
        rows.append(row)
        ema_str = f"{fid_ema:.3f}" if fid_ema is not None else "n/a"
        print(f"  |G|={bs} s{seed} (ep {ckpt.get('epoch')}): "
              f"FID_live={fid_live:.3f}  FID_ema={ema_str}  ({secs:.0f}s)")
        del model, fp
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Aggregate + paired deltas under each weighting.
    delta_live = paired_delta(rows, "fid_live")
    delta_ema = (paired_delta([r for r in rows if r["fid_ema"] is not None], "fid_ema")
                 if any(r["fid_ema"] is not None for r in rows) else None)

    def agg(key):
        out = {}
        for bs in (1, 2, 4):
            vals = [r[key] for r in rows
                    if r["block_size"] == bs and r.get(key) is not None]
            if vals:
                n = len(vals); mean = sum(vals) / n
                sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
                out[str(bs)] = {"n": n, "fid_mean": mean, "fid_std": sd}
        return out

    payload = {
        "config": {"ckpt_dir": args.ckpt_dir, "ckpt_glob": args.ckpt_glob,
                   "n_fid_samples": args.n_fid_samples,
                   "note": "fid_live = no-EMA (live U-Net weights); "
                           "fid_ema = EMA shadow (reproduces v3)."},
        "per_run": rows,
        "aggregate_live": agg("fid_live"),
        "aggregate_ema": agg("fid_ema"),
        "paired_delta_1v4_live": delta_live,
        "paired_delta_1v4_ema": delta_ema,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== VERDICT (Δ FID(1−4); positive = block |G|=4 WINS, v1 direction) ===")
    if delta_live:
        sgn = "BLOCK WINS  → favors (b) EMA artifact" if delta_live["mean_diff"] > 0 \
              else "BLOCK LOSES → favors (a) real ELBO–FID gap"
        print(f"  LIVE (no-EMA): Δ = {delta_live['mean_diff']:+.2f} "
              f"± {delta_live['sd_diff']:.2f} (n={delta_live['n']})   {sgn}")
    if delta_ema:
        print(f"  EMA  (v3 repro): Δ = {delta_ema['mean_diff']:+.2f} "
              f"± {delta_ema['sd_diff']:.2f} (n={delta_ema['n']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
