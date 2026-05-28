# PERFECT_REPO_PROPOSAL — what this folder contains

Drop-in replacements + a few new files for the v3 audit-of-audit. Drop the
contents of this folder into the repo root to apply.

## Files modified (overwrite the existing repo file)

| File | Why |
|---|---|
| `fldd/train.py` | + `EMA` class, `use_ema` context manager, EMA-aware `train_epoch`. + Seeded `Generator` in `compute_validation_elbo` for bit-reproducibility. Default `loss_form="true_elbo"`. |
| `train_mnist.py` | + `--ema_decay` flag. Saves three checkpoint kinds (`_best.pt` train-loss, `_valbest.pt` val-ELBO, `_final.pt`) every run. `--restore` controls which gets loaded back into the returned model. Default `loss_form="true_elbo"`. |
| `fldd/forward.py` | Added a docstring note that α_T pins at the structural 0.5 upper bound (the B1 fix only addresses the lower bound). |
| `REVIEW.md` | (i) **Corrected B5 gradient direction** (was inverted in v2). (ii) **Flagged B4 frozen-α `[0.06, 0.50]` as itself the v1 floor** — use the v2-learned T=2 schedule instead. (iii) Added the v3 revision note up top, plus **new item B18** (EMA missing) and **new §F.7** with the 4-cell pilot ablation. |
| `run_e2_fast.py` | + `--ema_decay`, `--save_prefix`, `--cell_name` flags. FID is scored on the returned (restored, EMA-loaded) model — no separate ckpt round-trip. |

## Files newly added

| File | Why |
|---|---|
| `recompute_elbo.py` | Ported from teammate. Analytically subtracts the parasitic `H[q]` from the v1 reported losses to produce corrected ELBOs without retraining. |
| `viz_samples.py` | Ported from teammate. Paper-grade qualitative sample figure (GT \| \|G\|=1 \| 2 \| 4) with shared `z_T`. Adapted for our `best/valbest/final` naming + EMA-aware loading. |
| `tests/test_schedule_sanity.py` | Hardened pytest version of teammate's `sanity_schedule.py`. Tests both `sigmoid_offset=-2` (floor saturates) and `sigmoid_offset=-6` (no effective floor), and the α_T = 0.5 upper-bound cap. |
| `scripts/run_chain_24h.sh` | The new 8-phase chain. Pilot (5 cells) → main n=3 → bs2 n=3 → extra n=6 seeds → FID MC → stats + figures + summary. Single invocation, ~22-27h queued. See below. |
| `scripts/pick_best_ablation.py` | Reads the 5 pilot ablation JSONs and prints the winner so the chain script can pick it up via `eval $(...)`. |
| `scripts/fid_mc.py` | Re-scores each ckpt with 3 different RNG seeds; reports sample-set sd alongside seed sd (REVIEW.md B9.ii). EMA-aware. |

## Files unchanged (no copies in this folder — keep what's already in the repo)

`fldd/blocks.py`, `fldd/unet.py`, `fldd/data.py`, `fldd/sample.py`,
`fldd/synthetic.py`, `fldd/block_analysis.py`,
`run_e3.py`, `run_e5_tc_decomp.py`, `run_e1.py`,
`merge_e2_stats.py`, `merge_e2_holm.py`, `merge_e4_stats.py`,
`viz_schedule.py`, `mnist_fd.py`, `evaluate_fid.py`,
`scripts/run_chain.sh` (kept for reference; the new `run_chain_24h.sh`
supersedes it), `scripts/rescore_best_pt.py`,
`tests/test_blocks.py`, `tests/test_forward.py`, `tests/test_stats.py`,
`tests/test_train.py`, `STATUS.md`, `RERUNS.md`, `CHANGELOG.md`,
`README.md`, `PROBLEMSETTING.md`.

## The 24-hour chain (scripts/run_chain_24h.sh)

**Single-invocation, ~10-hour default.** No follow-up scripts needed. The
chain queues just enough work to land near the 10h target, with opt-in
flags to extend if you want more.

| Phase | What | Default | Extension flag |
|---|---|---|---|
| **P1 PILOT** | **5-cell** ablation on bs4 seed 42, 50 epochs. Cells: `{ema_off, ema_on} × {select=train_loss, select=val_loss}` + a 5th cell with fixed v1 α's. All `sigmoid_offset=-6`, `loss_form=true_elbo`. | **ON** (~3h) | — |
| **P2 MAIN n=3** | Winning config × bs ∈ {1, 4} × seeds {42, 43, 44} × 100 epochs. | **ON** (~7.5h) | — |
| **P3 BS=2 n=3** | Same config × bs=2 × seeds {42, 43, 44}. Adds the 1→2→4 monotone story. | OFF | `--with_bs2` (+~3.5h) |
| **P4 EXTRA bs1/4** | Seeds {45, 46, 47} × bs ∈ {1, 4}. Brings n=6 to match v1's protocol. | OFF | `--with_extras` (+~7.5h) |
| **P5 EXTRA bs=2** | Seeds {45, 46, 47} × bs=2. | OFF | `--with_bs2 --with_extras` (+~3.5h) |
| **P6 FID MC** | Re-FID each winning ckpt 3× with different RNG seeds (REVIEW.md B9.ii). | OFF | `--with_fid_mc` (+~1.5h) |
| **P7 STATS+FIG** | Paired Holm 1v4 (+ 1v2, 2v4 if bs2). Schedule plot. `viz_samples.py`. `recompute_elbo.py` on v1 ckpts. | **ON** (~0.5h) | — |
| **P8 SUMMARY** | `results/v3/SUMMARY.json`. | **ON** (<1m) | — |

### Default run (no flags) ≈ **11 hours**

```bash
tmux new -s aml-chain
bash scripts/run_chain_24h.sh
# Ctrl+B D to detach.
```

Lands ~11h. Headline: n=3 paired comparison bs1 vs bs4 under the
pilot-winning config, plus all figures and the v1 corrected ELBOs.

### Extension presets

```bash
bash scripts/run_chain_24h.sh --with_bs2          # ~14.5h, adds 1→2→4 monotone
bash scripts/run_chain_24h.sh --with_extras       # ~18.5h, adds n=6 bs1/4
bash scripts/run_chain_24h.sh --with_bs2 --with_extras  # ~22h, n=6 across all bs
bash scripts/run_chain_24h.sh --max               # ~24h, everything on
```

The expected runtime is printed at startup so you know how long to expect.

### Safety properties

* **Single invocation.** No `--only=p1` followups; everything runs end-to-end.
* **Fallback winner.** If the pilot fails to identify a winner (e.g. all
  5 cells crashed), the chain proceeds with `ema_on + val_loss` rather
  than aborting — keeps the GPU productive.
* **Per-run timeout.** Each training run is capped at 2h via `timeout`,
  so a hung job doesn't eat the whole budget.
* **Idempotent.** SKIP guards on existing ckpts + JSONs. If the chain
  does crash and you restart it, it picks up where it left off.

### What we'll know at the end

* **From P1 winner (5-cell read)**: which v2 change is responsible for the absolute-FID regression.
  * EMA-on (cell 3 or 4) winner → standard diffusion EMA explains the FID gap; the v2 methodology was sound, the pipeline was just missing EMA.
  * EMA-off train-loss (cell 1) winner → the issue is the val-stopping criterion; keep train-loss selection.
  * EMA-off val-loss (cell 2) winner (= current behaviour) → it's the schedule itself. Cell 5 (v1-frozen-α + EMA + val) confirms this if it beats cell 4.
  * Cell 5 winning specifically → the learned v2 schedule is FID-suboptimal vs the v1-floor schedule. Real ELBO-FID gap finding.
* **From P2**: n=3 headline at the winning config, comparable to v1's
  numbers and teammate's quick sweep.
* **From P4**:
  * Schedule plot showing the post-fix learned schedule across seeds and
    block sizes (cross-replicates the teammate's reparam6 result).
  * Qualitative sample figure for the paper.
  * v1 corrected ELBOs from the existing 18 ckpts (no retraining).

### Strong-paper-claim path

Best case (EMA wins big): "v2 methodology is correct; absolute FID
matches or beats v1 once EMA is added; block advantage is preserved."
That's the cleanest publishable story.

Second-best (val-stopping is the issue): "the train-loss best-ckpt
selection was empirically right despite being methodologically loose;
we report both criteria for transparency."

Third (schedule itself is the issue): "the learned graded schedule is
ELBO-optimal but FID-suboptimal — illustrating a known ELBO-FID gap and
arguing for either fixed schedules or FID-based selection in
discrete diffusion." Still publishable as a methodology contribution.
