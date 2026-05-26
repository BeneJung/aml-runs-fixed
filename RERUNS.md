# GPU Reruns Plan (after P0 code fixes)

This document is for whoever has GPU access. The code fixes from the
methodology review are in; what remains is to run training and FID
to confirm the headline results are robust. Everything below is
**additive** — original commands still work and reproduce the existing
numbers exactly.

## What changed in the code

| File | Change | Bug ID |
|------|--------|--------|
| `fldd/forward.py` | `LearnedForwardProcess` now accepts `sigmoid_offset` (default −2.0, historical) and `fixed_alphas`. Old checkpoints load with `strict=True`. | B1, B4 |
| `fldd/train.py` | `compute_elbo_loss(..., loss_form=)` supports `"ce"` (historical) and `"true_elbo"` (strict KL). Added `compute_validation_elbo` (deterministic, sums over all t). | B5 |
| `fldd/data.py` | `get_binarized_mnist(..., val_fraction=, split_seed=)` for held-out validation split. | B6 |
| `train_mnist.py` | New flags: `--sigmoid_offset`, `--fixed_alphas`, `--loss_form`, `--val_fraction`, `--select_by`, `--val_samples_per_t`. Saves these into the checkpoint. | B1, B4, B5, B6 |
| `run_e2.py` | Forwards `--sigmoid_offset`, `--loss_form`, `--val_fraction`, `--select_by` to `run_mnist`. | same |
| `run_e4_t2_frozen.py` | NEW. Runs T=2 with a frozen schedule shared across `|G| ∈ {1, 4}`. | B4 |

CPU smoke tests passed. No GPU required so far.

## Sanity check before any expensive run (~30 s on GPU)

Verify that the historical defaults still reproduce the old run on a
single seed:

```bash
python train_mnist.py --T 4 --block_size 4 --seed 42 --epochs 1 \
    --save_dir /tmp/sanity_old
# expected: alphas near init values, recon loss in the right ballpark
```

Then verify the new parameterization moves the floor:

```bash
python train_mnist.py --T 4 --block_size 4 --seed 42 --epochs 1 \
    --sigmoid_offset -6.0 --save_dir /tmp/sanity_new
# expected console line: "alpha floor (parameterization): 0.001236"
```

## R1 — Schedule reparameterization (P0, ~6 GPU hours)

**Question:** Does the schedule actually learn away from the floor when
the structural lower bound is much smaller? This determines whether the
"FLDD learns a data-aware schedule" framing survives.

```bash
# 1. Retrain E2 at T=4 with the new parameterization, 3 seeds first
python run_e2.py --device cuda --epochs 100 \
    --seeds 42 43 44 --block_sizes 1 4 \
    --sigmoid_offset -6.0 \
    --loss_form true_elbo \
    --val_fraction 0.1 --select_by val_loss \
    --save_dir checkpoints_e2_floor6 \
    --gen_root fid_stats_e2_floor6 \
    --results_json results/results_e2_floor6.json

# 2. Visualize the learned schedule
python viz_schedule.py \
    --e2_dir checkpoints_e2_floor6 \
    --out_prefix figures/viz_schedule_e2_floor6 \
    --summary_json results/schedule_summary_floor6.json
```

**Decision criteria after R1:**

- **Schedules now diverge across runs / block sizes:** the FLDD framing
  is rescued. Make this the new default, retire `sigmoid_offset = -2.0`
  from the paper experiments, and report the new schedule plot.
- **Schedules still saturate at the new floor (~0.00124):** the
  optimizer truly wants α → 0 for early steps. Re-frame the paper as
  "block-factorized discrete diffusion with near-no-op early steps."
  Either is publishable.
- **Mixed (e.g. α_T moves but α_1..3 still saturate):** report the
  exact pattern; this is itself a finding about FLDD on this setup.

After R1, regardless of outcome, the headline FID claim needs to be
verified under the new schedule. If R1 changes the Δ FID by less than
~2 FID points the original headline stands; otherwise update the
table.

## R2 — Validation-based checkpoint selection (P0, ~0 extra GPU)

R1 already runs with `--val_fraction 0.1 --select_by val_loss`, so this
is bundled. To isolate the effect on the *existing* checkpoints, also
re-score `checkpoints_e2/*_best.pt` with FID (already done in
`results_e2_from_ckpts.json`) and compare to a `final.pt` re-score:

```bash
# Score the existing FINAL.pt checkpoints (selected by nothing — just
# the last epoch) and compare to the existing best.pt FIDs. If the gap
# is large, the headline depends on the selection rule and B6 is
# decisive.
python eval_e2_from_ckpts.py --device cuda \
    --ckpt_dir checkpoints_e2 \
    --results_json results/results_e2_from_final.json
# (NB: you'll need to point eval_e2_from_ckpts at *_final.pt instead of
# *_best.pt — either rename or pass a different glob. One-line patch.)
```

The point is to characterise how unstable the headline is to
checkpoint choice using the *current* runs, before paying for new ones.

## R3 — T=2 frozen-schedule rerun (P0, ~3 GPU hours)

**Question:** Is the 42-point Δ FID at T=2 due to the reverse head
(the variable of interest) or the schedule (a confound the README
already flagged)?

```bash
# Recommended: freeze to |G|=4's learned T=2 schedule (the more
# generous-to-baseline choice — gives bs1 every advantage)
python run_e4_t2_frozen.py \
    --device cuda --epochs 80 --seeds 42 43 44 \
    --schedule from_bs4_ckpt --bs4_ckpt_dir checkpoints_e4 \
    --loss_form true_elbo \
    --val_fraction 0.1 --select_by val_loss

# Cross-check with a neutral fixed schedule
python run_e4_t2_frozen.py \
    --device cuda --epochs 80 --seeds 42 43 44 \
    --schedule explicit --alphas 0.06 0.50 \
    --results_json results/results_e4_t2_frozen_explicit.json
```

**Decision criteria:**

- Frozen Δ FID similar to the original 42 (say within 10 FID): the
  reverse-head effect is real and the T=2 finding survives.
- Frozen Δ FID much smaller (say < 15): the original 42-point gap was
  mostly the schedule confound. Re-state the E4 finding accordingly.

## R4 — Bump E3 to all 6 seeds (P2, ~0 GPU, just CPU eval)

The bs4 checkpoints for seeds 45/46/47 already exist in
`checkpoints_e2`. `run_e3.py` globs all of them, so this is free:

```bash
python run_e3.py --device cuda  # already picks up all bs4_s*_best.pt
```

Then update the README's "3 seeds, 2048 test images" to "6 seeds".

## R4b — Vertical 2×1 block control (P2, ~3 GPU hours)

**Question (B11):** Does the 1×2 → 2×2 jump come from "capturing both
axes" or just from "larger blocks"? A 2×1 vertical control answers this
at the same parameter count as 1×2 horizontal.

```bash
python train_mnist.py --T 4 --block_size 2 --orientation vertical \
    --seed 42 --epochs 100 --device cuda \
    --sigmoid_offset -6.0 --loss_form true_elbo \
    --val_fraction 0.1 --select_by val_loss \
    --save_dir checkpoints_e2_vert
# repeat for seeds 43-47

# Then score with FID using evaluate_fid.py against the cached real dir.
```

**Decision criteria:** if vertical 2×1 ≈ horizontal 1×2 in FID, the
1×2 → 2×2 jump is mostly "bigger block." If they differ substantially,
axis matters and the paper should report both.

## R5 — Within- vs between-block TC decomposition (P2, ~0 extra GPU)

**Question (B13):** What fraction of the total cross-pixel TC is
within-block (absorbable by |G|=4) vs between-block (irreducible)?
This is post-hoc on existing |G|=4 ckpts.

```bash
python run_e5_tc_decomp.py --device cuda \
    --ckpt_dir checkpoints_e2 --block_size 4 \
    --n_images 512 --mc_samples 8 --n_pair_samples 64
```

Outputs `results/results_e5_tc_decomp.json` and
`figures/e5_tc_decomp.{png,pdf}`. The "within fraction" gives the
paper a direct quantitative claim: "block-factorization absorbs X% of
the cross-pixel TC at t=T."

## R6 — Secondary metric: MNIST-classifier Frechet distance (P1, ~1.5 GPU hour)

**Question (B9.i):** InceptionV3 on binarized MNIST is noisy. Report a
second metric using MNIST-specific features as a robustness check.

```bash
# 1. Train the small classifier once (~1 min on CPU)
python mnist_fd.py --train_classifier --epochs 3

# 2. Score every E2 best.pt with MNIST-FD
python mnist_fd.py --score_ckpts checkpoints_e2 --device cuda \
    --n_samples 10000 \
    --results_json results/mnist_fd_e2.json
```

If MNIST-FD ranks the block sizes the same way FID does, the headline
ordering is robust to feature-extractor choice. If it disagrees,
report both metrics and discuss.

## R7 — FID Monte-Carlo averaging (P1, ~1 GPU hour)

Re-score each E2 best.pt with 3 independent sample sets and report
sample-set variance separately from seed variance. One-shot:

```bash
for run in 0 1 2; do
    python eval_e2_from_ckpts.py --device cuda \
        --results_json results/results_e2_fid_run${run}.json
done
```

Merge externally. Expected: per-checkpoint FID sd ~0.3 across the 3
runs (matches the E2-vs-E4 drift we already saw).

## Order of operations

```
R1 (most informative; tells us what the paper is about)
 │
 ├─ if framing changes → rewrite intro/method, then continue
 │
R3 (cheap once R1 is done; uses R1 ckpts if needed)
 │
R2 (analytical, no new training)
 │
R4 (free, just rerun E3 with all 6 seeds)
 │
R5 (free, post-hoc TC decomposition on existing |G|=4 ckpts)
 │
R4b (optional, vertical-2x1 control — only if reviewers ask)
 │
R6 (cheap, secondary metric for robustness)
 │
R7 (last; only matters once headline is stable)
```

## What to put in the paper after the reruns

- A schedule-plot panel showing both old and new parameterization
  schedules side by side, with the floor lines drawn as dashed.
- The R3 result as a separate row in the E4 table: "T=2, schedule
  frozen, ...".
- Multiple-comparison correction stated explicitly (Holm).
- Selection rule stated explicitly ("best.pt by val-ELBO on held-out
  5–10% split").
- All `loss_form="true_elbo"` numbers comparable to literature ELBO;
  if mixing both, label clearly.
- Acknowledge in the limitations section that the original alpha-floor
  artefact existed and was found during methodology audit; cite the
  R1 reparameterization as the fix.

## What to *not* do (yet)

- Don't change `block_size=2` to a 2×1 variant yet (B11) — the 1×2
  story is fine for the current version, the 2×1 control is for v2.
- Don't add KID / precision-recall (B9.i) — pick this up only if a
  reviewer asks.
- Don't yet attempt the head-to-head against the 22.31 prior baseline
  (B3) — settle the internal story first.

## Estimated end-to-end GPU time

Core (P0): R1 ~6 h + R3 ~3 h = 9 GPU hours.
Plus P1/P2: R7 ~1 h + R6 ~1.5 h = 2.5 hours.
Plus optional R4b vertical control ~3 h.
Total: ~12-15 GPU hours for a stable, paper-ready set of numbers.

## What changed in the code (extended list)

In addition to B1/B4/B5/B6 (see "What changed in the code" at top):

| Bug ID | File | Change |
|--------|------|--------|
| B2 | `fldd/synthetic.py`, `run_e1.py` | Renamed `pixel_factorized_tv_floor` → `unconditional_pixel_marginal_tv_floor`; old name kept as deprecated alias. Reference output text now says "NOT a floor for FLDD." |
| B8 | `merge_e2_stats.py` | Added `holm_bonferroni()` helper. New script `merge_e2_holm.py` runs all 3 pairwise comparisons (1v2, 1v4, 2v4) and applies Holm across the family. |
| B10 | `merge_e4_stats.py` | Bootstrap CIs at n<6 now labelled "descriptive" in both text and JSON output. |
| B11 | `fldd/blocks.py`, `fldd/unet.py`, `fldd/train.py`, `fldd/sample.py`, `train_mnist.py` | Added `orientation="horizontal"\|"vertical"` for `block_size=2`. Same parameter count, different axis. |
| B13 | `run_e5_tc_decomp.py` (NEW) | Within- vs between-block TC decomposition from existing |G|=4 ckpts. |
| B9.i | `mnist_fd.py` (NEW) | Secondary Frechet-distance metric using MNIST-classifier features. |
| B16 | `README.md` | E2 SDs corrected to match JSON values (2.88 / 3.69 not 2.96 / 3.71). |
| backward compat | `run_e2.py`, `run_e4.py`, `run_e3.py`, `eval_e2_from_ckpts.py`, `viz_schedule.py` | All `LearnedForwardProcess` constructions read `sigmoid_offset` from the ckpt with a HISTORICAL_OFFSET fallback for old ckpts. |
