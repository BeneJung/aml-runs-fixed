"""Retroactively recompute the true ELBO of v1 (pre-audit) checkpoints by
subtracting the parasitic H[q] term from the reported CE-form loss.

The pre-fix training loss was `CE = KL + H[target]`. H[target] depends
only on the forward alphas (not the reverse model), so it can be
subtracted analytically without retraining:

    parasitic_per_image = 784 * sum_{j=0}^{T-2} H_b(alpha_j)   (nats)

H_b is the binary entropy. Block size does not matter: the entropy of a
product distribution is the sum of its marginal entropies, and there are
784 pixels either way. (At t=1 the target is delta(x), so its entropy is
zero and the sum stops at j = T-2.)

    corrected ELBO = reported loss - parasitic

This script reads two of the existing v1 result files plus
`schedule_summary.json` (which has per-ckpt alphas), matches them by
(T, block_size, seed), and writes `results/results_elbo_corrected.json`.

Ported from the teammate's branch (PERFECT_REPO_PROPOSAL audit). Lightly
adapted to:
  * accept additional source JSONs via `--sources`,
  * tolerate either `best_loss` / `loss` / `final_loss` keys (the v1
    per-run schema drifted across files),
  * write a human-readable summary alongside the JSON.
"""

import argparse
import json
import math
import os

N_PIX = 784


def hb(a):
    a = min(max(a, 1e-12), 1 - 1e-12)
    return -(a * math.log(a) + (1 - a) * math.log(1 - a))


def parasitic(alphas):
    T = len(alphas)
    # H[delta(x)] = 0 at t=1, so the sum stops at j = 0..T-2
    return N_PIX * sum(hb(alphas[j]) for j in range(0, T - 1))


def add_losses(loss_table, path, T_fixed=None):
    with open(path) as f:
        d = json.load(f)
    for r in d.get("per_run", []):
        T = r.get("T", T_fixed)
        L = r.get("best_loss")
        if L is None:
            L = r.get("loss")
        if L is None:
            L = r.get("final_loss")
        if T is None or L is None:
            continue
        loss_table[(int(T), int(r["block_size"]), int(r["seed"]))] = float(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule_summary",
                    default="results/schedule_summary.json")
    ap.add_argument("--sources", nargs="+",
                    default=["results/results_e4.json",
                             "results/results_e2_merged.json",
                             "results/results_e2_bs2.json"])
    ap.add_argument("--source_T_fixed", nargs="+", default=[None, 4, 4])
    ap.add_argument("--out", default="results/results_elbo_corrected.json")
    args = ap.parse_args()

    T_fixeds = []
    for v in args.source_T_fixed:
        if isinstance(v, str) and v.lower() in ("none", "null", ""):
            T_fixeds.append(None)
        else:
            try:
                T_fixeds.append(int(v))
            except (TypeError, ValueError):
                T_fixeds.append(None)
    if len(T_fixeds) < len(args.sources):
        T_fixeds += [None] * (len(args.sources) - len(T_fixeds))

    loss = {}
    for src, Tf in zip(args.sources, T_fixeds):
        if not os.path.exists(src):
            print(f"  [skip] missing source {src}")
            continue
        add_losses(loss, src, T_fixed=Tf)

    if not os.path.exists(args.schedule_summary):
        raise FileNotFoundError(
            f"missing {args.schedule_summary} — run viz_schedule.py first to "
            f"dump per-ckpt alphas."
        )
    with open(args.schedule_summary) as f:
        sched = json.load(f)

    rows = []
    seen = set()
    for e in sched.get("per_ckpt_e2", []) + sched.get("per_ckpt_e4", []):
        key = (int(e["T"]), int(e["block_size"]), int(e["seed"]))
        if key in seen:
            continue
        seen.add(key)
        par = parasitic(e["alphas"])
        L = loss.get(key)
        rows.append({
            "T": e["T"], "block_size": e["block_size"], "seed": e["seed"],
            "parasitic_Hq": par,
            "reported_loss": L,
            "corrected_elbo": (L - par) if L is not None else None,
            "corrected_elbo_per_step": ((L - par) / e["T"]) if L is not None else None,
            "Hq_fraction": (par / L) if L else None,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    hdr = (f"{'T':>3} {'bs':>3} {'seed':>4} {'reported':>10} "
           f"{'parasitic':>10} {'corrected':>10} {'per-step':>9} {'H[q]%':>6}")
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["T"], x["block_size"], x["seed"])):
        if r["reported_loss"] is None:
            continue
        print(f"{r['T']:>3} {r['block_size']:>3} {r['seed']:>4} "
              f"{r['reported_loss']:>10.2f} {r['parasitic_Hq']:>10.2f} "
              f"{r['corrected_elbo']:>10.2f} "
              f"{r['corrected_elbo_per_step']:>9.2f} "
              f"{100 * r['Hq_fraction']:>5.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
