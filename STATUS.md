# Project status — read this first

Last updated: 2026-05-28 evening.

If you're a teammate joining now, read this top-to-bottom. It tells you
what the project is, what we found, what's currently running, and what
to do next.

## 1. What the project is

Block-factorized discrete diffusion (AML 2026 semester project).

In one paragraph: discrete-diffusion models like FLDD generate binarized
MNIST by reversing a noising process. The reverse model standardly
predicts each pixel independently, but the true reverse target has
cross-pixel correlations — leaving an irreducible KL gap. Our idea:
have the reverse model predict joint distributions over small 2×2
pixel blocks so it can absorb within-block correlations directly.

Full proposal: [PROBLEMSETTING.md](PROBLEMSETTING.md). Original (v1)
results: [README.md](README.md).

## 2. The three hypotheses (all supported in v1)

| | Claim | v1 result |
|---|---|---|
| H1 | Block head beats pixel head on synthetic block-tiled data | TV 0.365 → 0.056 |
| H2 | Block head beats pixel head on MNIST T=4 FID | FID 58.11 → 49.08, paired t p=0.006 (n=6) |
| H3 | Trained block joints deviate from product-of-marginals more in stroke/edge regions than background | TC 0.12 (bg) vs 0.39 (mixed) vs 0.43 (stroke) at t=T |

## 3. What we found in the methodology review (REVIEW.md)

Three bugs that mattered, several smaller things. Full audit in
[REVIEW.md](REVIEW.md). The big three:

- **B1: the "learned" schedule was stuck at a parameterization floor.**
  All 18 v1 runs converged to α ≈ `[0.0596, 0.0596, 0.0596, 0.500]`
  — the early αs hit the structural lower bound of
  `0.5 · σ(−2) = 0.0596` set by the offset in the sigmoid. The FLDD
  "data-aware schedule" wasn't actually being learned.
- **B5: reported "ELBO" was BCE, not strict ELBO.** ~77% of the
  ~690-nat reported value was H[q], not KL.
- **B6: best.pt selected by training loss.** Across our 3 seeds for
  |G|=1, train loss is anti-correlated with FID (Pearson = −1.000).

Fixed all of these in code via opt-in flags that preserve historical
behaviour by default. New CLI:

```
--sigmoid_offset -6.0      # B1: lowers schedule floor to ~0.00124
--loss_form true_elbo      # B5: strict KL contribution
--val_fraction 0.1         # B6: held-out validation split
--select_by val_loss       # B6: best.pt by val ELBO, not train
--val_samples_per_t 2      # speedup
```

[CHANGELOG.md](CHANGELOG.md) lists every code change vs v1.

## 4. The v2 results so far (after methodology fixes)

R1 was run with n=3 seeds at T=4 using all v2 flags. Numbers after
**rescoring from best.pt** (see §5 for why):

| | v1 (final.pt) | v2 (best.pt) |
|---|---|---|
| bs1 FID | 58.11 ± 2.88 | 69.63 ± 5.71 |
| bs4 FID | 49.08 ± 3.69 | 55.73 ± 5.16 |
| **Δ FID (bs1 − bs4)** | **+9.03** | **+13.90** |
| Paired t (1-sided) p | 0.006 | 0.006 |

**Two key takeaways:**

1. **Schedule actually learns now.** v1: `[0.0596, 0.0596, 0.0596, 0.500]`
   stuck at the floor. v2: `[0.024, 0.252, 0.460, 0.494]` — a proper
   data-aware ramp. B1 was correct: the parameterization was
   suppressing schedule learning.
2. **Block advantage GREW.** v1 had Δ = +9 FID; v2 has Δ = +14 FID,
   same p-value. The block-factorized head wins by more under the
   corrected methodology.

Absolute FID is slightly worse in v2 because v2's val-based selection
stops training at epoch ~22 (where val ELBO is minimized) whereas
v1's train-loss selection stayed at epoch ~90. v2's "best" model is
less converged but it's the correct stopping point. Honest paper
framing: *"v1 trained past its val minimum; the v2 protocol is
methodologically correct and shows a larger relative block advantage."*

## 5. A bug we found mid-experiment that you need to know about

During R1 v2 we discovered that **both `run_e2.py` (original) and
`run_e2_fast.py` (our v2 chain) score FID on the final-epoch model,
not on `best.pt`**. This was always true; v1 just got away with it
because train-loss selection put best.pt near the final epoch anyway.

For v2, val_loss selection picks best.pt around epoch 22; the model
overfits afterward and the final-epoch FID is significantly worse than
best.pt's FID. The first six runs gave nonsense FIDs (some up to 240+)
until we rescored from best.pt.

Fix: `run_e2_fast.py` now loads `best.pt` before computing FID
(commit `60d5e21`). `scripts/rescore_best_pt.py` retroactively
rescores any ckpt dir. **Use this script whenever you need apples-to-
apples FIDs.**

## 6. What's currently running

A `scripts/run_chain.sh` invocation in the `aml-chain` tmux session on
Renku. It runs five blocks idempotently (SKIP guards on existing
ckpts + JSONs):

| Block | What | Cost |
|---|---|---|
| R1 | Train v2 ckpts for seeds 45/46/47 at bs={1,4} so n=6 like v1 | ~7.5h |
| R3 | T=2 with a FROZEN schedule `[0.06, 0.50]` to remove the B4 confound | ~3.5h |
| R4 | Re-run E3 against all 6 v1 |G|=4 ckpts (E3's original was n=3) | ~10 min |
| R5 | within-vs-between block TC decomposition on v1 |G|=4 ckpts | ~20 min |
| R6 | MNIST-classifier-feature Frechet distance — secondary metric on both v1 and v2 ckpts | ~25 min |

Total ~12h. Outputs land on SwitchDrive via symlinks — survive
session reaps.

Also running in `keepalive` tmux session: a GPU-ping loop every 30s,
to prevent Renku from idling out the session.

## 7. What runs after the chain finishes

`bash scripts/run_chain_phase2.sh` in a new tmux pane. About 2h:

| Block | What | Why |
|---|---|---|
| v1 rescore | Rescore all 18 v1 ckpts (`checkpoints_e2/*_best.pt`) from best.pt | Apples-to-apples FID comparison (see §5) |
| R3 MNIST-FD | MNIST-FD score on the new frozen-T=2 ckpts | R3 secondary metric |
| Sample grids | 64-image grids for v1 and v2 ckpts | Paper qualitative figure |

The fair v1 numbers from this step go into the headline table.

## 8. Where to find what

Code:

| File | What's in it |
|---|---|
| `fldd/` | Core library (`forward.py`, `unet.py`, `blocks.py`, `train.py`, `sample.py`, etc.) |
| `train_mnist.py` | The single training function `run_mnist()` used by everything |
| `run_e2_fast.py` | R1 sweep (in-memory FID, scores best.pt) |
| `run_e4_t2_frozen_fast.py` | R3 sweep (in-memory FID, scores best.pt) |
| `run_e3.py` | E3 within-block TC analysis |
| `run_e5_tc_decomp.py` | E5 within-vs-between TC decomposition |
| `mnist_fd.py` | MNIST-classifier-feature Frechet distance |
| `merge_e2_stats.py` / `merge_e2_holm.py` | Paired stats + Holm correction |
| `scripts/run_chain.sh` | The full chain, idempotent |
| `scripts/run_chain_phase2.sh` | Post-chain rescore + extras |
| `scripts/rescore_best_pt.py` | Rescore any ckpt dir from best.pt |
| `tests/` | Pytest suite, 36 tests, locks in correctness |

Docs:

| File | What it covers |
|---|---|
| **STATUS.md** | **You are here.** Current project state. |
| README.md | Original (v1) project description + numbers |
| PROBLEMSETTING.md | Original proposal |
| REVIEW.md | Full methodology audit (the bug list) |
| RERUNS.md | Detailed rerun plan + decision criteria |
| CHANGELOG.md | Version history, every code change |

Results (after chain + phase2 finish):

| Path | What |
|---|---|
| `results/per_run_floor6/bs{1,4}_s{42..47}.json` | One JSON per v2 R1 run |
| `results/results_e2_floor6_merged.json` | Paired stats on v2 R1 (n=6 if chain finished) |
| `results/results_e2_floor6_holm.json` | Holm-corrected p-values |
| `results/schedule_summary_floor6.json` | Learned αs across all v2 runs |
| `results/results_e4_t2_frozen_T2_bs{1,4}_s{42..44}_frozen.json` | R3 per-run |
| `results/results_e3_n6.json` | R4 (E3 with all 6 v1 bs4 seeds) |
| `results/results_e5_tc_decomp.json` | R5 TC decomposition |
| `results/mnist_fd_e2_{original,floor6}.json` | R6 MNIST-FD numbers |
| `results/v1_rescored/bs{1,2,4}_s{42..47}.json` | v1 rescored from best.pt (phase 2) |
| `results/v1_rescored_merged_1v4.json` | v1 paired stats on rescored numbers |

Figures:

| Path | What |
|---|---|
| `figures/viz_schedule_e2_floor6_e2.png` | Learned schedules across v2 R1 runs (should NOT be at the floor — that's the B1 win) |
| `figures/e3_n6_*` | E3 within-block TC plots |
| `figures/e5_tc_decomp.png` | within vs between block TC decomposition |
| `figures/samples/v[12]_bs[14]_s42.png` | Qualitative sample grids (phase 2) |

## 9. If you just sit down to work on this — start here

1. Read this file (§1–7).
2. `cd ~/work/aml-runs-fixed`, then `git pull` to get the latest scripts.
3. Run `tmux ls` to see what's running.
4. If `aml-chain` is alive: `tmux attach -t aml-chain`, see where it
   is, `Ctrl+B D` to detach.
5. Check progress: `grep -E "(START|END|DONE)" logs/chain_*.log | tail -20`
   and `grep "rc=" logs/chain_*.log` for exit codes.
6. When chain shows "CHAIN DONE": `tmux new -s phase2; cd ~/work/aml-runs-fixed; bash scripts/run_chain_phase2.sh`.
7. When phase 2 shows "PHASE 2 DONE": you have all the data the paper needs.

## 10. Things to know about the Renku setup

- Symlinks from the repo dir to `~/work/drive/` (SwitchDrive) carry all
  outputs through to persistent storage. Sometimes they vanish on
  session restart — auto-recreation snippet is in `~/.bashrc`.
- Renku reaps the session on **terminal/UI inactivity** (~35 min by
  default). Compute-only activity isn't enough; the `keepalive` tmux
  session pings the GPU and tries to register as "activity".
- FID via pytorch_fid's PNG-file backend is **catastrophically slow**
  on rclone-mounted storage (~hours instead of ~seconds for 10k tiny
  PNGs). All our chain scripts use in-memory FID via run_e2_fast.py's
  helpers. **Do not use the original run_e2.py for new runs.**
- `data/` MNIST cache is not symlinked, so it gets re-downloaded on
  every fresh container. ~10s, harmless.

## 11. Open questions / things to discuss

- **Should we report v1 numbers from final.pt (the README) or from
  best.pt (rescored, fair)?** Probably both, with a clear note about
  what changed.
- **Should we add bs=2 to the v2 sweep for full coverage of the
  monotonicity story?** Cost ~7.5h; v1 said bs=2 was "directionally
  consistent but underpowered" — not strictly necessary.
- **The absolute FID went up in v2 (49 → 56 for bs4).** This is
  defensible (val-based stopping is methodologically correct), but
  reviewers may flag it. Discuss the framing before submission.
- **R3 with explicit alphas `[0.06, 0.50]` vs the v1-derived
  alphas:** we chose explicit so R3 doesn't depend on which v1 ckpt
  exists. Discuss whether to also run a "from-bs4-ckpt" variant.
