# Project analysis + final-12h compute plan

Last updated: 2026-05-29 (written against the v3 chain results in STATUS.md
and the result JSONs verified below). Goal of the next 12 GPU-hours, per the
team decision: **settle the science** — decide between interpretation (a) and
(b) of the v3 FID reversal as definitively as the compute allows.

This document has three parts:

1. **Foundation** — what we built on (FLDD + the prior TC-regularization project).
2. **Our approach and where we stand** — block-factorized reverse head; the
   v1 → audit → v3 arc; the H2 reversal and the exact open question.
3. **The 12-hour plan** — a GPU-serial timeline, the decision tree, and the
   runnable scripts that ship with this document.

A running theme: every headline number below was re-verified against the raw
JSONs (`results/…`, `NEW REPO UPDATED REPO/results/…`) before being repeated.
Where the local snapshot and the Renku/SwitchDrive state diverge, that is
flagged explicitly — it matters for what is runnable in the next 12 h.

---

## Part 1 — The foundation we build on

### 1.1 FLDD (Forward-Learned Discrete Diffusion)

Discrete diffusion generates data **x** ∈ {1,…,K}^D by learning to reverse a
corruption process. FLDD's distinguishing move is that the **forward (noising)
schedule is learned jointly with the reverse model** — a "data-aware" schedule
— which is what lets it generate in very few steps (T = 4 here). Our binarized
MNIST setting is K = 2, D = 784.

The structural cost of FLDD's data-awareness is a **factorization gap**. The
true reverse target q(z_s | z_t), averaged over the unknown clean **x**,
develops cross-pixel correlations. The standard reverse model p_θ predicts
each pixel **independently**, so it cannot represent those correlations. The
mismatch is measured by the **Total Correlation (TC)** of the aggregated
posterior and is *irreducible* no matter how much reverse-model capacity you
add — because the factorized form itself is the bottleneck, not the parameter
count.

### 1.2 The prior DL-course project (the thing we improve on)

A previous DL-course project attacked the gap **from the forward side**: it
added a TC penalty to the objective (L = ELBO + λ·TC), with entropic-OT
coupling and a REINFORCE baseline. Reported result: **FID 40.80 → 22.31** on
MNIST at T = 4. Its structural problems — which motivate our approach — were:

- The TC estimator is **high-variance at D = 784**, so the regularization
  signal is noisy.
- Penalizing TC pushes q_φ toward a **data-blind schedule**, which directly
  fights FLDD's core benefit.
- **Extreme λ-sensitivity**: FID 22.31 at λ = 10⁻³ vs 52.57 at λ = 10⁻².

The takeaway we inherited: reducing TC in the *forward* process is the wrong
lever — it degrades the very schedule that makes FLDD good.

> ⚠️ **Comparison caveat (carry this into any talk).** The 22.31 FID from the
> prior project is **not** directly comparable to our numbers: different
> architecture, training budget, and possibly binarization. REVIEW.md B3
> downgrades this to a presentation problem, not a methodology bug — but a
> reviewer who skims will read "22.31 → 49/34" as a regression. Either
> enumerate the protocol differences or frame strictly as "vs. our own
> pixel-factorized baseline under matched protocol."

---

## Part 2 — Our approach and where we stand

### 2.1 The idea: absorb local TC in the *reverse* model

Instead of reducing TC in the forward process, we **absorb local TC in the
reverse model** by predicting a joint distribution over small pixel blocks.
For a block partition G = {G_1, …, G_B}:

```
p_θ(z_s | z_t) = Π_b p_θ(z_s^{G_b} | z_t)
```

each block emitting a categorical over K^|G_b| joint states. The KL loss
decomposes cleanly:

```
KL[q || p_θ] = TC_between (irreducible) + Σ_b KL[q^{G_b} || p_θ^{G_b}] (reducible)
```

A 2×2 block on binarized MNIST is a **16-state softmax** — cheaper than a
typical classification head. The block head absorbs *within-block* TC by
construction, shrinking the irreducible gap from TC_total to TC_between.

The mechanism is subtle and worth stating precisely for the talk: each
*individual* training target q_φ(z_s^G | z_t, **x**) is itself factorized
(the coupling is element-wise). But **different x produce different factorized
targets for the same z_t**; averaged over training, the block head learns the
*data-averaged* target — a mixture of products, which is a genuine joint that
captures within-block correlation. A pixel-factorized head is forced to
collapse that mixture to a product of marginals.

### 2.2 The three hypotheses and the v1 evidence

| | Claim | v1 result | Status |
|---|---|---|---|
| **H1** | Block head beats pixel head on synthetic block-tiled data | block-TV 0.365 → 0.056 (≈6.5×), paired t p = 0.0072 | **solid, controlled** |
| **H2** | Block head beats pixel head on MNIST T=4 FID | FID 58.11 → 49.08, Δ = 9.03, paired t p = 0.006, n = 6 | **the headline — now contested (see 2.4)** |
| **H3** | Trained block joints deviate from product-of-marginals more in stroke/edge than background | within-block TC 0.12 (bg) vs 0.39 (mixed) vs 0.43 (stroke) at t = T | **solid, mechanism evidence** |

H1 and H3 are robust and largely compute-independent. **The entire tension is
in H2.** E4 (stretch) added FID-vs-T ∈ {2,4,8,16}, with the block advantage
largest at T = 2 (Δ ≈ 42) — but that row has a known schedule confound (B4).

### 2.3 The methodology audit (REVIEW.md) — three bugs that move numbers

The audit is the project's strongest asset: it is genuinely rigorous and it
found real problems. The three that change conclusions:

- **B1 — the "learned" schedule was stuck at a parameterization floor.** All
  18 v1 runs converged to α ≈ [0.0596, 0.0596, 0.0596, 0.50]. 0.0596 is
  exactly `0.5·σ(−2)`, the structural floor set by the sigmoid offset. The
  "data-aware schedule" was **not being learned at all** — fatal to the FLDD
  framing as originally told. Verified analytically and to 4+ digits against
  `schedule_summary.json`.
- **B5 — the reported "ELBO" was cross-entropy, not strict KL.** CE = KL +
  H[q]; H[q] depends on the schedule, not on θ. Verified from
  `results_elbo_corrected.json`: parasitic H[q] ≈ **531 nats** out of the
  reported ~690 → **~77 %** of the "ELBO" was entropy, not model fit. The
  corrected per-image ELBO for |G|=1 is ≈ 158 nats. (Also: the parasitic
  gradient pushed α *down*, the same direction as the B1 floor — the two bugs
  compounded.)
- **B6 — best.pt was selected by *training* loss.** For |G|=1 across 3 seeds,
  train-loss and FID are **anti-correlated (Pearson = −1.000)**: the
  lowest-train-loss checkpoint had the *worst* FID. So the selection rule
  itself can move the headline.

All three were fixed behind opt-in flags (`--sigmoid_offset -6.0`,
`--loss_form true_elbo`, `--val_fraction 0.1 --select_by val_loss`) that
preserve historical behaviour by default. A 4th item, **B18 — no EMA**, was
added in the v3 audit-of-audit: the pipeline trained plain Adam with no
exponential moving average, which is non-standard for diffusion (DDPM-class
methods treat decay-0.9999 EMA as table stakes).

### 2.4 v3: the reversal — and the exact open question

The v3 chain (`scripts/run_chain_24h.sh`, ran 2026-05-29, 8 h 41 m) applied
**all** fixes at once: reparam offset −6, true-ELBO loss, val-loss selection,
**and EMA**, with a 5-cell pilot choosing the config. Headline
(`results/v3/SUMMARY.json`):

| `|G|` | v1 FID (train-sel, n=6) | **v3 FID (n=3, paired)** | v1 corrected ELBO | v3 ELBO |
|---|---|---|---|---|
| 1 (pixel) | 58.11 ± 2.88 | **33.76 ± 0.50** | 159.31 | ~96.4 |
| 4 (2×2) | 49.08 ± 3.69 | 42.57 ± 1.06 | 125.20 | ~85.4 |
| **Δ FID (1 − 4)** | **+9.03 (block wins)** | **−8.80 (block LOSES)** | | |

Two facts coexist after the fixes:

- **FID reversed.** Per-seed Δ FID = (−7.98, −8.23, −10.20); every seed flips.
  Paired t = −12.55, p = 0.006. This is not noise. **The pixel head now makes
  better-looking samples than the block head.**
- **ELBO ordering preserved.** v3 |G|=4 still has ~11 nats lower ELBO than
  |G|=1 (paired t = 49.4, p = 0.0002). **The block head still fits the data
  better.** This is expected — it is strictly more expressive.

So we have a clean **ELBO–FID gap**: block fits better, samples worse. The
question that decides the entire paper framing is *why*:

> **(a) Real finding.** Methodology-correct training reveals an intrinsic
> ELBO–FID gap. v1's H2 was an artifact of B1/B5/B6 + no EMA. Honest
> reposition: *"block factorization improves ELBO but not FID; it demonstrates
> the ELBO–FID gap in discrete diffusion."*
>
> **(b) EMA–block interaction artifact.** EMA-averaging the categorical
> 16-state head softens the t = 1 argmax in a way the 1-channel pixel head
> doesn't suffer. If so, **without EMA the block head still wins** and the
> original H2 holds.

### 2.5 What the existing data already says about (a) vs (b)

There is already a strong hint for **(b)**, which I verified directly:

- **Teammate's `reparam6_quick` (no EMA, val-sel, reparam, 30 ep, seeds
  42–43):** FID means |G|=1 = **71.83**, |G|=2 = 58.11, |G|=4 = **54.61**.
  **Block still wins by ~17 FID points, monotone in block size**, same
  direction as v1. (`NEW REPO UPDATED REPO/results/results_e2_reparam6_quick.json`.)
- **v3 chain (EMA, 100 ep):** block *loses* by ~9.
- The two differ in EMA **and** epochs **and** the exact reparam form — so it
  is a strong hint, **not proof**. n = 2 seeds, 30 epochs is underpowered.

A second, sharper observation about the mechanics makes the decisive test
**cheap**:

> 🔑 **EMA never feeds back into training.** In `fldd/train.py`, `ema.update()`
> runs *after* `optimizer.step()` and writes only to its own shadow — it never
> touches the live parameters or the gradient. Therefore the **live U-Net
> weights saved in each v3 checkpoint are exactly what a no-EMA run with the
> same seed would have produced** at that epoch. Each v3 checkpoint stores both
> the live weights (`ckpt["model"]`) and the EMA shadow (`ckpt["ema"]`).
> **Re-scoring FID on the live weights is ~15 min and needs no retraining** —
> and it isolates the EMA effect on the *same* trajectory.

The one residual confound the cheap re-score does *not* remove: the checkpoint
*epoch* was selected by val-ELBO measured **on the EMA weights**, so a true
no-EMA run might stop at a slightly different epoch. The full EMA-off retrain
closes that gap. The plan does both, so the conclusion rests on three
independent legs.

### 2.6 ⚠️ State-of-the-repo caveat (load-bearing for the plan)

This local snapshot is at the **v3 code** state (HEAD = `a8a0214 "v3 audit EMA
+ …"`), but the **v3 *outputs* are not in it**: there is no `results/v3/`, no
`results/pilot_ablation/`, no `checkpoints_v3/`, no
`figures/viz_schedule_v3_*` locally. Per STATUS.md and the team's
confirmation, those live on **Renku/SwitchDrive**, where the 12 h will run.
**Everything in the plan assumes you run on that machine** (`REPO=~/work/aml-runs-fixed`,
`checkpoints_v3/` present). Step 0 verifies this before anything expensive.

---

## Part 3 — The 12-hour plan: settle (a) vs (b)

**Objective:** decide, with paper-grade confidence, whether the block head's
FID loss in v3 is intrinsic (a) or an EMA artifact (b). Secondary: leave the
ELBO–FID gap characterized either way.

**Design.** Build the clean 2×2 that v3 never completed — {EMA on, EMA off} ×
{|G|=1, |G|=4} at matched protocol — and read off whether **EMA flips the sign
of the block advantage Δ FID(1−4)**. v3 already supplies the EMA-on, n=3 cell.
We add EMA-off (retrain) and, on the existing v3 checkpoints, the near-free
live-weight re-score. Three independent legs:

| Leg | What it isolates | Cost | Decides |
|---|---|---|---|
| **D1** live-vs-EMA re-score of v3 ckpts | EMA *weight-averaging* effect, same trajectory | ~0.5 h | (a)/(b) provisionally, first |
| **D4** EMA-off retrain, n=3, 100 ep | EMA *training* effect, clean val-selection | ~7.5 h | (a)/(b) definitively |
| **D6** EMA-off extra seed(s) | statistical power on the headline cell | ~2.5 h **per seed** | strength of the verdict |

> **Budget reality (one GPU, ~75 min/run — your v3 chain's own figure):** D1 +
> D4(n=3) + D3 + D5 ≈ **8.3 h**, leaving ~3.7 h. That fits **one** extra
> EMA-off seed (n=4, ≈10.8 h total) — **not** a full n=6, which is ~15.5 h and
> belongs in a second session. Run n=3 as the must; add seed 45 if the early
> verdict is clean and time remains.

Supporting, ~free: **D2** pilot-cell inspection, **D3** sample grids +
block-joint entropy (the *mechanism* of b), **D5** ELBO–FID decoupling (falls
out of D4's logs).

### 3.1 GPU-serial timeline (single GPU, ~12 h)

```
[0.00–0.10h]  Step 0  Verify Renku state: checkpoints_v3/ + results/v3/ exist
[0.10–0.50h]  D1      fid_no_ema.py: live vs EMA FID on the 6 v3 ckpts   ← provisional verdict
[0.50–0.60h]  D2      pick_best cell inspection (CPU, free)              ← why EMA won the pilot
   ── decision peek: does live-weight FID rank |G|=4 < |G|=1? ──
[0.60–8.10h]  D4      EMA-off n=3 sweep: bs∈{1,4} × seeds 42–44 × 100 ep ← the clean control
[8.10–10.6h]  D6      EMA-off seed 45 (bs1+bs4) → n=4 headline  ← if D4 confirms & time remains
[10.6–11.1h]  D3      viz_samples (patched): EMA-on grids, bs1 vs bs4
[11.1–11.2h]  D5      ELBO–FID decoupling table from D4 logs (CPU, free)
[11.2–11.4h]  wrap    merge stats, write SETTLED.json, dump the 2×2 table
[ buffer ]            ~0.6h slack; full n=6 (seeds 46,47) = a separate session
```

CPU-only steps (schedule plots, `recompute_elbo.py`, `merge_e2_stats.py`) run
in a second pane any time and don't consume the GPU budget.

### 3.2 The decision tree (what each outcome means)

Read **Δ FID(1−4)** in each cell (positive = block wins, the v1 direction):

- **D1 live-weight Δ > 0 (block wins without the EMA shadow).** → Strongly
  favors **(b)**. EMA weight-averaging is hurting the block head. Proceed with
  D4 as confirmation; spend the tail (D6) on one extra EMA-off seed (n=4) for
  more power, and finish n=6 in a second session if the restored H2 is the
  headline.
- **D1 live-weight Δ < 0 (block still loses on live weights).** → Favors
  **(a)**; the gap is not just the EMA shadow. D4 becomes the decisive test of
  whether a *fully* EMA-free training run (clean val-selection) still loses. If
  D4 also shows Δ < 0, the ELBO–FID gap is **real** — reposition the paper on
  (a), and reallocate D6 → **R5 within/between-block TC decomposition** to
  quantify the gap mechanism instead of buying seeds.
- **D4 disagrees with D1.** → The effect is in the *epoch selection* (val-ELBO
  measured on EMA weights), not the weights themselves. Report both; the
  honest story is "EMA changes which checkpoint looks best, and that flips the
  FID ranking." Still resolves to (b)-flavoured but with a precise mechanism.

**Go/no-go for restoring H2:** declare (b) and restore H2 only if **both** D1
(live) **and** D4 (EMA-off retrain) show Δ FID(1−4) > 0, ideally with the n=6
top-up clearing p < 0.05 paired. Otherwise the honest headline is the ELBO–FID
gap (a).

### 3.3 Exact commands

Step 0 — verify state (do not skip; the plan is void without these):

```bash
export REPO=~/work/aml-runs-fixed && cd "$REPO"
ls checkpoints_v3/bs1_s42_valbest.pt checkpoints_v3/bs4_s42_valbest.pt
ls results/v3/SUMMARY.json results/pilot_ablation/cell_*.json
git pull   # get fid_no_ema.py, run_settle_science.sh, and the sample.py patch
```

D1 — the decisive cheap re-score (ships with this doc as `scripts/fid_no_ema.py`):

```bash
python scripts/fid_no_ema.py \
    --ckpt_dir checkpoints_v3 --ckpt_glob '*_valbest.pt' \
    --n_fid_samples 10000 \
    --out results/v3/fid_no_ema.json
# prints, per |G|, FID(live) and FID(ema), and Δ FID(1−4) under each weighting
```

D4 + D6 — the EMA-off sweep (one variable changed vs v3: **no `--ema_decay`**).
The whole chain, including D1/D2/D5 and optional D6, is wrapped in
`scripts/run_settle_science.sh`:

```bash
tmux new -s settle
export REPO=~/work/aml-runs-fixed && cd "$REPO"
bash scripts/run_settle_science.sh                    # D1+D2+D4(n=3)+D3+D5   (~8.3h)
EXTRA_SEEDS="45" bash scripts/run_settle_science.sh   # also seed 45 → n=4   (~10.8h)
# Ctrl+B D to detach. Idempotent: re-run later with more EXTRA_SEEDS for n=6.
```

The bare D4 sweep, if you want to run it by hand:

```bash
python run_e2_fast.py --device cuda --T 4 --epochs 100 \
    --seeds 42 43 44 --block_sizes 1 4 \
    --sigmoid_offset -6.0 --loss_form true_elbo \
    --val_fraction 0.1 --select_by val_loss --val_samples_per_t 1 \
    --save_dir checkpoints_emaoff --save_prefix '' \
    --results_json results/v3/emaoff_n3.json
    # NOTE: no --ema_decay → EMA off. Everything else identical to the v3 main sweep.
```

### 3.4 Rigor guardrails (the AML bar)

- **Match the pilot winner.** Before D4, confirm from `results/pilot_ablation/`
  which cell v3's main sweep inherited (STATUS implies *EMA-on, val-sel,
  learned schedule* — i.e. **not** the v1-frozen-schedule cell). D4 must match
  that config in everything **except** EMA. `scripts/run_settle_science.sh`
  prints the pilot table at the top so this is visible.
- **Pair correctly.** Δ FID(1−4) is paired *by seed*. Compare EMA-off vs
  EMA-on with a paired test on the per-seed Δs, not on the marginals.
- **Don't reuse `[0.06, 0.50]` as a "controlled" schedule** (B4/REVIEW §F.7):
  it *is* the v1 floor and reintroduces the artifact. Not needed for this
  plan, but flagged for any T=2 follow-up.
- **Optional-stopping (B7).** When reporting the n=3 vs n=6 EMA-off result,
  state plainly that n=6 was a pre-planned power top-up, not seed-fishing.
- **Three legs, one story.** Only claim the verdict where D1 (live re-score),
  D4 (EMA-off retrain), and the existing `reparam6_quick` hint agree. If they
  diverge, report the divergence — that is itself the finding.
- **n = 10k FID has ~0.3 MC noise (B9.ii).** Treat sub-0.5-point FID
  differences as ties; the block effects here (~9–17) are far above that.

### 3.5 What ships with this document

- `scripts/fid_no_ema.py` — **NEW.** Re-scores any v3 ckpt dir twice (live
  weights, then EMA shadow), paired, and prints Δ FID(1−4) under each
  weighting. This is D1, the decisive cheap test. Modeled exactly on the
  existing `scripts/rescore_best_pt.py` (same model/forward reconstruction).
- `scripts/run_settle_science.sh` — **NEW.** Orchestrates D1 → D2 → D4 →
  D3 → D5 (+ D6 via `EXTRA_SEEDS="45 …"`), idempotent (SKIP guards), never
  aborts mid-chain, prints the final 2×2 Δ FID table and writes
  `results/v3/SETTLED.json`.
- `fldd/sample.py` — **PATCHED** (backward-compatible): `sample()` now accepts
  `generator=` and `z_init=`, fixing the `viz_samples.py` crash noted in
  STATUS §0.4 so D3 (sample grids, paired noise across block sizes) runs.

> These files exist in this local repo. **They must be committed and pushed so
> Renku's `git pull` picks them up** before Step 0.

---

## Appendix — numbers re-verified for this document

| Claim | Source JSON | Verified |
|---|---|---|
| v1 E2 headline 58.11 / 49.08, Δ=9.03, t=3.88, p₁=0.0058, n=6 | `results/results_e2_merged.json` | ✓ exact |
| B5: parasitic H[q] ≈ 531 nats, ~77 % of ~690; corrected ELBO ≈158 (|G|=1) | `NEW REPO UPDATED REPO/results/results_elbo_corrected.json` | ✓ exact |
| Schedule learns off the floor: selected α ≈ [0.023, 0.236, 0.454, 0.493] | `…/results_e2_reparam6_quick.json` | ✓ exact |
| reparam6_quick (no EMA): |G|=1 71.83, |G|=2 58.11, |G|=4 54.61 — block wins | `…/results_e2_reparam6_quick.json` | ✓ exact |
| Checkpoint stores live `model` + `forward` + `ema` shadow; EMA is post-step only | `fldd/train.py`, `train_mnist.py` | ✓ read |
| v3 reversal headline (33.76 / 42.57, Δ=−8.80) | `results/v3/SUMMARY.json` | ⚠ **not in local snapshot — on Renku** |
