"""Confirm the EMA pitfall MECHANISM: does EMA flatten the block head's
final-step softmax (making the argmax less decisive)?

For each reverse step t, we run the live model's reverse trajectory and, at the
*same* latent z_t, evaluate the output distribution under BOTH the live weights
and the EMA-shadow weights. The only thing that changes is the weight set, so
any entropy gap is attributable to EMA.

Metrics (per step, mean over blocks and images):
  * block head (|G|>1): H = entropy of the 16-way softmax (nats; max = ln16 = 2.77);
                        margin = top1 - top2 probability (1.0 = fully decisive).
  * pixel head (|G|=1): H = Bernoulli entropy (nats; max = ln2 = 0.69);
                        margin = |2p - 1| (1.0 = fully decisive).

Mechanism is CONFIRMED if, at the final step t=1, the block head has
  H_ema > H_live  and  margin_ema < margin_live
(EMA softmax is flatter / less decisive) and this gap is larger for the block
head than for the pixel head.

Read-only, ~30 s on GPU. Usage (Renku):
    python scripts/ema_entropy_probe.py \
        --ckpt_dir checkpoints_v3 --ckpt_glob '*_valbest.pt' \
        --n 512 --out results/v3/ema_entropy_probe.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.blocks import block_indices_to_pixels
from fldd.train import EMA, use_ema

CKPT_RE = re.compile(r"(?:T(\d+)_)?bs(\d+)_s(\d+)_")


def measure(logits, block_size):
    """Return (mean_entropy_nats, mean_decisiveness_margin)."""
    if block_size == 1:
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        H = -(p * p.log() + (1 - p) * (1 - p).log())          # Bernoulli entropy
        margin = (2 * p - 1).abs()                            # |2p-1|
    else:
        p = F.softmax(logits, dim=1)                          # (B, K^|G|, Hb, Wb)
        H = -(p.clamp_min(1e-9).log() * p).sum(dim=1)
        top2 = p.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]                      # top1 - top2
    return float(H.mean()), float(margin.mean())


@torch.no_grad()
def probe_ckpt(path, n, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    T, bs = ckpt["T"], ckpt["block_size"]
    off = ckpt.get("sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET)
    fixed = ckpt.get("fixed_alphas")

    model = UNet(channels=(32, 64, 128), block_size=bs).to(device).eval()
    model.load_state_dict(ckpt["model"])
    fp = LearnedForwardProcess(T=T, sigmoid_offset=off, fixed_alphas=fixed).to(device).eval()
    fp.load_state_dict(ckpt["forward"])

    ema = None
    if ckpt.get("ema"):
        ema = EMA(model, decay=float(ckpt["ema"].get("decay", 0.9999)))
        ema.load_state_dict(ckpt["ema"], device=device)

    z = torch.bernoulli(0.5 * torch.ones(n, 1, 28, 28, device=device))
    rows = []
    for t in range(T, 0, -1):
        tb = torch.full((n,), t - 1, device=device, dtype=torch.long)

        logits_live = model(z, tb)
        H_live, m_live = measure(logits_live, bs)
        if ema is not None:
            with use_ema(model, ema):
                logits_ema = model(z, tb)
            H_ema, m_ema = measure(logits_ema, bs)
        else:
            H_ema = m_ema = float("nan")
        rows.append({"t": t, "H_live": H_live, "H_ema": H_ema,
                     "margin_live": m_live, "margin_ema": m_ema})

        # advance z along the LIVE trajectory (mirrors fldd/sample.sample)
        if bs == 1:
            probs = torch.sigmoid(logits_live)
            z = torch.bernoulli(probs) if t > 1 else (probs > 0.5).float()
        else:
            probs = F.softmax(logits_live, dim=1)
            if t > 1:
                B, nS, Hb, Wb = probs.shape
                flat = probs.permute(0, 2, 3, 1).reshape(-1, nS)
                idx = torch.multinomial(flat, 1).squeeze(-1).reshape(B, 1, Hb, Wb)
            else:
                idx = logits_live.argmax(dim=1, keepdim=True)
            z = block_indices_to_pixels(idx, bs)
    return bs, T, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints_v3")
    p.add_argument("--ckpt_glob", default="*_valbest.pt")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/v3/ema_entropy_probe.json")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)))
    if not paths:
        raise SystemExit(f"no {args.ckpt_glob} under {args.ckpt_dir}")

    # average the per-step curves across seeds, grouped by block size
    by_bs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_ckpt = []
    for path in paths:
        bs, T, rows = probe_ckpt(path, args.n, args.device)
        per_ckpt.append({"ckpt": os.path.basename(path), "block_size": bs, "rows": rows})
        m = CKPT_RE.search(os.path.basename(path))
        seed = m.group(3) if m else "?"
        for r in rows:
            for k in ("H_live", "H_ema", "margin_live", "margin_ema"):
                by_bs[bs][r["t"]][k].append(r[k])
        t1 = [r for r in rows if r["t"] == 1][0]
        print(f"  |G|={bs} s{seed}: t=1  H_live={t1['H_live']:.3f} H_ema={t1['H_ema']:.3f}"
              f"  margin_live={t1['margin_live']:.3f} margin_ema={t1['margin_ema']:.3f}")

    summary = {}
    for bs, tdict in by_bs.items():
        summary[str(bs)] = {}
        for t, kd in sorted(tdict.items()):
            summary[str(bs)][str(t)] = {k: (sum(v) / len(v)) for k, v in kd.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": {"n": args.n, "ckpt_dir": args.ckpt_dir},
               "summary_by_bs_t": summary, "per_ckpt": per_ckpt},
              open(args.out, "w"), indent=2)

    print("\n=== FINAL STEP (t=1) — mean across seeds ===")
    print(f"{'|G|':>4} {'H_live':>8} {'H_ema':>8} {'ΔH(ema-live)':>13} "
          f"{'marg_live':>10} {'marg_ema':>9} {'Δmargin':>9}")
    for bs in sorted(summary, key=int):
        s = summary[bs].get("1")
        if not s:
            continue
        dH = s["H_ema"] - s["H_live"]
        dM = s["margin_ema"] - s["margin_live"]
        print(f"{bs:>4} {s['H_live']:>8.3f} {s['H_ema']:>8.3f} {dH:>+13.3f} "
              f"{s['margin_live']:>10.3f} {s['margin_ema']:>9.3f} {dM:>+9.3f}")
    print("\nMechanism CONFIRMED if the block head (|G|=4) shows ΔH > 0 and Δmargin < 0,")
    print("and larger in magnitude than the pixel head (|G|=1).")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
