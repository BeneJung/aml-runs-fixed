"""Per-T paired statistics on E4 FID results (|G|=1 vs |G|=4).

Reuses the stats primitives from merge_e2_stats.py. With n=3 per T,
non-parametric tests bottom out at p=0.125 (one-sided sign / Wilcoxon
support is too small to reach 0.05). The paired t-test (df=2) and the
bootstrap CI are the only sub-0.05 routes; we report all four for parity
with E2 and let the reader weigh the small-n caveat.
"""

import json

from merge_e2_stats import bootstrap_ci, paired_t, sign_test, wilcoxon


# Below this n, the percentile bootstrap CI on the mean paired diff is
# effectively a min/max readout of the data (there are only n^n distinct
# resamples), so the 2.5/97.5 quantiles round to the extrema. We still
# compute it for parity with E2 but label it "descriptive" so the table
# isn't read as an inferential interval.
BOOTSTRAP_INFERENTIAL_THRESHOLD = 6


def main():
    with open("results/results_e4.json") as f:
        d = json.load(f)

    by_T = {}
    for r in d["per_run"]:
        by_T.setdefault(r["T"], []).append(r)

    out = []
    for T in sorted(by_T):
        by_seed = {}
        for r in by_T[T]:
            by_seed.setdefault(r["seed"], {})[r["block_size"]] = r["fid"]
        pairs = []
        diffs = []
        for s in sorted(by_seed):
            if 1 in by_seed[s] and 4 in by_seed[s]:
                a, b = by_seed[s][1], by_seed[s][4]
                pairs.append({"seed": s, "bs1": a, "bs4": b, "diff": a - b})
                diffs.append(a - b)
        ttest = paired_t(diffs)
        w = wilcoxon(diffs)
        st = sign_test(diffs)
        ci = bootstrap_ci(diffs)
        print(f"\n=== T={T}  (n={len(diffs)}) ===")
        for p in pairs:
            print(f"  seed {p['seed']}: bs1={p['bs1']:.4f}  bs4={p['bs4']:.4f}  "
                  f"Δ={p['diff']:+.4f}")
        print(f"  mean Δ = {ttest['mean_diff']:+.4f}   sd Δ = {ttest['sd_diff']:.4f}")
        print(f"  paired t = {ttest['t']:.3f}   p(one-sided) = {ttest['p_one_sided']:.4f}")
        if w:
            print(f"  Wilcoxon p(one-sided) = {w['p_one_sided']:.4f}")
        print(f"  sign = {st['n_positive']}/{st['n']}   p(one-sided) = {st['p_one_sided']:.4f}")
        is_descriptive = len(diffs) < BOOTSTRAP_INFERENTIAL_THRESHOLD
        ci_label = ("descriptive — n too small for inferential bootstrap"
                    if is_descriptive else "bootstrap 95% CI")
        print(f"  bootstrap 95% CI on Δ = [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
              f"   [{ci_label}]")
        ci["is_descriptive"] = bool(is_descriptive)
        ci["note"] = (
            f"With n={len(diffs)} only {len(diffs)**len(diffs)} distinct "
            "resamples exist; the 2.5/97.5 percentiles collapse to data "
            "extrema and the interval should not be read as inferential."
            if is_descriptive else
            "Standard percentile bootstrap, n_boot=20000."
        )
        out.append({
            "T": T, "pairs": pairs, "t_test": ttest,
            "wilcoxon": w, "sign_test": st, "bootstrap_ci": ci,
        })

    with open("results/results_e4_paired.json", "w") as f:
        json.dump({"by_T": out}, f, indent=2)
    print("\nwrote results/results_e4_paired.json")


if __name__ == "__main__":
    main()
