"""E2 multi-pair stats with Holm-Bonferroni family-wise correction.

The original README reports three pairwise FID comparisons (|G|=1 vs 2,
1 vs 4, 2 vs 4) on the same six paired seeds, each with a 1-sided
paired t-test. That's a 3-test family and the per-comparison p-values
should be Holm-Bonferroni adjusted to control the family-wise type-I
error at the stated alpha. This script does the adjustment once across
all three pairs and prints a clean table.

Usage
-----

    python merge_e2_holm.py \\
        --sources results/results_e2_from_ckpts.json \\
                  results/results_e2_extra_bs1.json \\
                  results/results_e2_extra_bs1_s45.json \\
                  results/results_e2_extra_bs4.json \\
                  results/results_e2_bs2.json
"""

import argparse
import json

from merge_e2_stats import (
    load_runs,
    collect_pairs,
    paired_t,
    wilcoxon,
    sign_test,
    bootstrap_ci,
    holm_bonferroni,
)


PAIRS = [(1, 2), (1, 4), (2, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=str,
                        default="results/results_e2_holm.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    all_runs = []
    for path in args.sources:
        for r in load_runs(path):
            r["source"] = path
            all_runs.append(r)

    # dedupe by (block_size, seed); first source wins
    seen = set()
    deduped = []
    for r in all_runs:
        key = (r["block_size"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    all_runs = deduped
    print(f"loaded {len(all_runs)} runs after dedup")

    per_pair = {}
    for (a, b) in PAIRS:
        pairs = collect_pairs(all_runs, a, b, "fid")
        if not pairs:
            print(f"  skip {a}v{b}: no overlapping seeds")
            continue
        diffs = [pa - pb for _, pa, pb in pairs]
        t = paired_t(diffs)
        w = wilcoxon(diffs)
        s = sign_test(diffs)
        ci = bootstrap_ci(diffs)
        per_pair[f"{a}v{b}"] = {
            "n_pairs": len(diffs),
            "mean_diff": t["mean_diff"],
            "sd_diff": t["sd_diff"],
            "paired_t_p_one_sided": t["p_one_sided"],
            "wilcoxon_p_one_sided": w["p_one_sided"] if w else None,
            "sign_p_one_sided": s["p_one_sided"],
            "bootstrap_ci": ci,
        }

    # Holm correction over the family of paired-t one-sided p-values
    p_dict = {
        label: per_pair[label]["paired_t_p_one_sided"]
        for label in per_pair
    }
    holm = holm_bonferroni(p_dict, alpha=args.alpha)
    for label, info in holm.items():
        per_pair[label]["holm_adjusted_p"] = info["adjusted"]
        per_pair[label]["holm_reject"] = info["reject"]

    # also Holm over Wilcoxon
    p_dict_w = {
        label: per_pair[label]["wilcoxon_p_one_sided"]
        for label in per_pair
        if per_pair[label]["wilcoxon_p_one_sided"] is not None
    }
    if p_dict_w:
        holm_w = holm_bonferroni(p_dict_w, alpha=args.alpha)
        for label, info in holm_w.items():
            per_pair[label]["holm_adjusted_wilcoxon_p"] = info["adjusted"]
            per_pair[label]["holm_reject_wilcoxon"] = info["reject"]

    print(f"\n=== Holm-Bonferroni family across {len(per_pair)} pairs "
          f"(alpha = {args.alpha}) ===\n")
    hdr = (f"{'pair':>6} | {'n':>2} | {'mean Δ':>8} | "
           f"{'sd Δ':>6} | {'t p 1s':>9} | {'Holm':>9} | reject | "
           f"{'Wilc p 1s':>9} | {'sign':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, info in per_pair.items():
        rej = "YES" if info["holm_reject"] else " no"
        wp = (f"{info['wilcoxon_p_one_sided']:.4f}"
              if info["wilcoxon_p_one_sided"] is not None else "n/a")
        sign_n = "{}/{}".format(
            sum(1 for d in [
                d for _, a, b in collect_pairs(
                    all_runs, *(int(x) for x in label.split('v')), 'fid'
                ) for d in [a - b]
            ] if d > 0),
            info["n_pairs"],
        )
        print(
            f"{label:>6} | {info['n_pairs']:>2} | "
            f"{info['mean_diff']:+8.3f} | "
            f"{info['sd_diff']:6.3f} | "
            f"{info['paired_t_p_one_sided']:.4f}    | "
            f"{info['holm_adjusted_p']:.4f}    |   {rej}  | "
            f"{wp:>9} | {sign_n:>5}"
        )

    payload = {
        "alpha": args.alpha,
        "pairs_tested": [f"{a}v{b}" for a, b in PAIRS],
        "per_pair": per_pair,
        "sources": args.sources,
    }
    def _to_jsonable(o):
        # numpy bools / floats not native-JSON-serializable
        try:
            import numpy as np
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
        except ImportError:
            pass
        raise TypeError(f"{type(o).__name__} not JSON-serializable")

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
