# Changelog

All notable changes to the codebase, grouped by version and by the bug
ID from [REVIEW.md](REVIEW.md). Defaults are preserved for backward
compatibility unless explicitly noted, so existing checkpoints continue
to reproduce historical numbers.

## v2 (post-methodology audit)

A full audit ([REVIEW.md](REVIEW.md)) identified one critical
parameterization artefact, a checkpoint-selection issue, and a handful
of presentation / statistics items. The fixes below are all in;
GPU-side reruns are tracked in [RERUNS.md](RERUNS.md).

### Added

- `fldd/forward.py`: `LearnedForwardProcess` now accepts
  `sigmoid_offset` (default −2.0, historical) and `fixed_alphas`
  (B1, B4). Old checkpoints load with `strict=True`. New attribute
  `alpha_floor` exposes the parameterization lower bound.
- `fldd/data.py`: `get_binarized_mnist(..., val_fraction=, split_seed=)`
  for a held-out validation split (B6).
- `fldd/train.py`:
  - `compute_elbo_loss(..., loss_form=)` supports `"ce"` (historical)
    and `"true_elbo"` (strict KL, B5).
  - `compute_validation_elbo()` — deterministic ELBO estimator that
    sums over all t and averages multiple z_t samples per t, suitable
    for val-loss best-checkpoint selection.
  - Both functions accept `orientation` for the 2×1 vertical variant.
- `fldd/blocks.py`: every block utility now accepts
  `orientation="horizontal"|"vertical"`; vertical only affects
  `block_size=2` and produces a 2×1 vertical block with the same
  parameter count (B11).
- `fldd/unet.py`: `UNet(..., orientation=)`; `BlockOutputHead` picks
  the right conv kernel shape per orientation.
- `train_mnist.py` CLI: `--sigmoid_offset`, `--fixed_alphas`,
  `--loss_form`, `--val_fraction`, `--select_by`, `--val_samples_per_t`,
  `--orientation`. Saved into the checkpoint metadata.
- `run_e2.py`, `run_e4.py`: forward the new flags to `run_mnist`.
- `run_e4_t2_frozen.py` (NEW): T=2 rerun with the forward schedule
  frozen across `|G| ∈ {1, 4}` (B4 fix).
- `run_e5_tc_decomp.py` (NEW): within- vs between-block TC
  decomposition for trained `|G|=4` checkpoints (B13).
- `mnist_fd.py` (NEW): Frechet distance using a small MNIST-classifier
  feature extractor as a secondary metric (B9.i). Includes
  `--train_classifier` to (re)train + cache the extractor.
- `merge_e2_stats.py`: `holm_bonferroni()` helper for family-wise
  step-down correction.
- `merge_e2_holm.py` (NEW): runs all three pairwise comparisons
  (1v2, 1v4, 2v4) at once and applies Holm across the family (B8).
- `run_e1.py`: paired t-test, Wilcoxon, sign test, bootstrap CI on
  Δ = TV(|G|=1) − TV(|G|=4) across seeds.
- `tests/` (NEW): pytest suite covering block reshape correctness for
  all (block_size, orientation) modes, parameterization floor math,
  CE-vs-true-ELBO theta-gradient identity, fixed-alphas invariance,
  checkpoint round-trip in both old and new format, Holm correction
  identities. 36 tests, all passing.
- `REVIEW.md`, `RERUNS.md`, `CHANGELOG.md` (this file) at repo root.

### Changed

- `README.md`: corrected E2 main-table SDs (2.96/3.71 → 2.88/3.69 to
  match the JSON values; B16). Added Holm-adjusted p-value column to
  the E2 pairwise-comparisons table. Added E1 paired-t result.
  Reframed the 0.72 reference as the *unconditional* pixel-marginal TV,
  with an explicit note that it is **not** a floor for pixel-factorized
  FLDD. New "Methodology fixes (v2)" section documents every new flag.
- `fldd/synthetic.py`: rewrote the module docstring; renamed
  `pixel_factorized_tv_floor` → `unconditional_pixel_marginal_tv_floor`.
  The old name remains as a deprecated alias for one release.
- `run_e1.py`, `merge_e4_stats.py`: bootstrap CIs are now labelled as
  "descriptive" when n<6 (B10), with a `is_descriptive` flag in the
  JSON output.
- `fldd/unet.py`: removed the dead `in_conv` skip from
  `skip_channels` and from the runtime `skips` stack. Pure code-clarity
  fix; trained weights and forward computation are unchanged (verified
  with existing E2 checkpoints).
- `fldd/forward.py`: deleted the unused `q_posterior` method; replaced
  with an explanatory comment about FLDD's non-Markovian structure
  (B17). Cleaned up the stream-of-consciousness inline derivation.

### Removed

- `LearnedForwardProcess.q_posterior` (unused, see Changed).
- `fldd/forward.py` lines 80–104 of inline "let me reconsider"
  derivation comments (replaced with a single clean note).

### Fixed

- B1: schedule was saturating against a parameterization floor of
  0.0596 in every E2 run. Now configurable via `--sigmoid_offset`.
- B5: loss reported as "ELBO" was actually CE = KL + H[q]; the H[q]
  term inflates the value by ~77% under the historical schedule and
  leaks a parasitic gradient into the schedule update. New
  `--loss_form true_elbo` removes both effects.
- B6: best.pt was selected by training loss, which is anti-correlated
  with FID for |G|=1 across our 3 seeds. New `--val_fraction +
  --select_by val_loss` enables proper held-out selection.
- B8: pairwise p-values across 3 tests were uncorrected. Holm
  correction now reported; 1v4 and 2v4 remain significant (adjusted
  p = 0.012, 0.004 respectively).
- B16: README SDs corrected to match JSON-derived values.

### Backwards compatibility

- All defaults are historical. Running the existing CLI commands with
  no new flags reproduces the published numbers exactly.
- Old checkpoints (created before v2) load with `strict=True` because
  `sigmoid_offset` and `fixed_alphas` are constructor arguments, not
  parameters. Checkpoint metadata now stores `sigmoid_offset` so
  loading scripts pick the right value automatically; missing fields
  default to the historical −2.0.
- `pixel_factorized_tv_floor` kept as an alias; will be removed in v3.

### Known unfixed items (require GPU / team input)

- **B1 verification rerun** — schedule reparameterization needs to be
  actually trained to confirm the schedule learns away from the new
  (lower) floor. RERUNS.md §R1.
- **B3** — gap to the prior 22.31 FID baseline cited in
  PROBLEMSETTING.md. Needs a research decision (drop the comparison,
  or run the prior method under the new protocol).
- **B4 T=2 rerun** — code is in (`run_e4_t2_frozen.py`); needs GPU
  to actually train. RERUNS.md §R3.
- **B6 rerun** — code is in; needs GPU. RERUNS.md §R2.
- **B7** — optional-stopping risk on the E2 n=3→n=6 seed expansion.
  Needs the team to clarify the decision protocol; cannot be inferred
  from the data alone.
- **B9.ii** — FID Monte-Carlo noise; resample each ckpt 3× and report
  sample-set sd separately. RERUNS.md §R7.
- **B11 vertical control rerun** — code is in (`--orientation
  vertical`); RERUNS.md §R4b. Optional unless reviewers ask.
- **B14** — E3 with 6 seeds; just rerun (code already globs). RERUNS.md §R4.
- **B15** — T=4 row in E4 uses 100 epochs while the others use 80.
  Needs retraining at matched epochs.

## v1 (initial submission)

The original AML 2026 project state. Reproduces the numbers in the
existing `results/*.json`. See git history for the per-file changes
before the v2 audit.
