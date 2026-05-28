# Methodology Review — Block-Factorized Discrete Diffusion

Rigorous audit of correctness, logic, and methodology with an eye toward
paper-quality publication. Items are graded P0 (must fix), P1 (should fix),
P2 (nice-to-have).

**Revision note (v2):** I re-checked every item with code/data evidence after
being asked "are you sure?" Concrete revisions: **B6 upgraded P1→P0**
(train-loss is *anti-correlated* with FID for |G|=1 across the seeds, so the
checkpoint selection rule actually does change the headline numbers);
**B2 downgraded P0→P1** (the 0.72 number is correct as the *unconditional*
floor, the issue is only the wording in the docstring); **B3 reframed P0→P1**
(it's a presentation/comparison concern, not a methodology bug); **B12
downgraded P1→P2** (the argmax behaviour is each model's native operation,
not an unfair asymmetry); **B7 explicitly downgraded** to "risk, not defect"
(I can't prove optional-stopping from the data alone). Original P0 items
B1 and B4 are confirmed. A new section [F. Next steps](#f-next-steps)
proposes a concrete plan.

**Revision note (v3 — 2026-05-29 audit-of-audit):** Two things found wrong
in the v2 revision itself, plus one new finding:

- **B5 gradient-direction claim was inverted** (v2 said "parasitic toward
  *higher*-entropy α"; the right direction is toward *lower*-entropy α,
  consistent with the v1 floor saturation). Rewritten below.
- **B4 frozen-α value `[0.06, 0.50]` is itself the v1 floor** — using it
  as the "schedule-controlled" baseline reintroduces the artefact B1
  exists to remove. Recommendation rewritten below.
- **B6 fix is methodologically correct but trades absolute FID for the
  fix.** Across the v2 R1 sweep and the independent reparam6_quick branch,
  val-ELBO best-checkpoint at epoch ~22 gives FID ~70 / ~56 for |G|=1/4
  vs the historical v1 train-best-at-epoch-90 of 58 / 49. The block
  *advantage* grows (Δ ≈ 9 → ~14 FID points), but absolute FID
  regresses. Possible root causes (one or more):
    1. Validation ELBO is a poor proxy for FID in diffusion (well-known).
    2. The pre-audit pipeline never used EMA — diffusion models almost
       always do, and EMA-smoothed weights typically improve FID by
       5–15 points. **New item B18 below.**
    3. The learned graded schedule is *worse for sampling* than the
       degenerate v1 schedule (a real ELBO/FID gap finding).
  A four-cell ablation (`{ema on/off} × {select=train, select=val}`,
  all with `loss_form=true_elbo`, `sigmoid_offset=-6`) is the single
  experiment that disambiguates these. Tracked in §F.7.

**New item B18 (P0, NEW).** **EMA missing from the training pipeline.**
The pre-audit repo trains plain Adam with no exponential moving average
on the U-Net weights. This is non-standard for diffusion models —
DDPM-class methods report EMA decay 0.9999 as table-stakes, and FID
without EMA is typically several points worse. Fix is ~30 lines in
`fldd/train.py`. Tracked in §F.7.

## TL;DR

The core idea is sound and the central empirical finding (block-factorized
reverse head improves FID over pixel-factorized at T=4) is real and replicated
across six paired seeds with appropriate statistics. The implementation is
mostly correct — block reshape/indexing, within-block TC, and the paired
statistics all check out. The headline weakness is that the "learned forward
schedule" is not actually being learned: it has collapsed to a numerical
parameterization floor in all 18 runs, which undercuts the FLDD framing and
also creates a clean confound at T=2 in E4. A handful of secondary
methodological points (analytic-floor claim, BCE-vs-KL gradient, best-by-train-loss
checkpoint selection, optional-stopping risk on seeds, multiple comparisons,
FID protocol) need to be cleaned up before the work can sit comfortably
next to published baselines.

---

## A. What's correct

These were checked and pass:

1. **Block reshape / indexing.** `pixels_to_blocks` uses
   `x.reshape(B, 1, H//2, 2, W//2, 2)`. Because the original tensor is
   row-major contiguous, this is equivalent to a 2×2-spatial decomposition:
   `((h_blk·2 + h_intra)·W) + (w_blk·2 + w_intra) == h·W + w` with
   `h = h_blk·2 + h_intra`, `w = w_blk·2 + w_intra`. The state encoding
   `b00·8 + b01·4 + b10·2 + b11` is consistent across `pixels_to_blocks`,
   `block_indices_to_pixels`, `compute_block_target`, and
   `_pixel_bit_table` in `block_analysis.py`.

2. **Within-block TC computation.** Marginalization (`joint_to_pixel_marginals`)
   and product-of-marginals reconstruction (`factorize_from_marginals`) match
   the standard definitions. The reported analytic sanity check — the
   maximally coupled 50/50 mixture of all-0 and all-1 gives
   `TC = 3·log 2 ≈ 2.079` — is correct (KL of two equiprobable points against
   the uniform-over-16 product is `log(0.5/0.0625) = log 8 = 3·log 2`).

3. **Paired statistics.** `merge_e2_stats.py` correctly implements a paired
   t-test (df = n−1), Wilcoxon signed-rank, sign test, and a percentile
   bootstrap CI on the mean paired difference. The one-sided p-values and
   the bootstrap interval recover the README values to four digits.

4. **Synthetic dataset construction.** Peaks `(0, 6, 9, 15)` correspond to
   all-0, the two 2×2 checkers, and all-1; each pixel marginal in the data
   is exactly 0.5 by symmetry. The data has only within-block coupling (no
   cross-block correlation), so it is the cleanest possible probe of local
   TC absorption.

---

## B. Logical / methodological issues

### P0 — Must address before paper

**B1. The "learned" forward schedule is hitting a parameterization floor in
every run.** `LearnedForwardProcess.get_alphas` returns

```
alphas = 0.5 · sigmoid( cumsum(softplus(logits)) − 2 )
```

The minimum value any α_t can take is `0.5·σ(−2) = 0.0596009…`. Every
"early" α reported in `schedule_summary.json` is `0.0596…` to four+ digits.
That is the floor itself, reached when `softplus(logit_t) → 0` (i.e.
`logit_t → −∞`). The "schedule collapse" the README flags is therefore not
a property of FLDD on binarized MNIST — it is the optimizer driving the
early logits against a structural lower bound, with α_T saturating at the
parameterization upper bound of 0.5 by an analogous mechanism. Consequences:

- The headline claim "all FID comparisons are at the same forward process" is
  technically correct but uninformative — the forward process isn't being
  learned at all in any of these runs.
- The whole FLDD framing of "data-aware schedule learned jointly with the
  reverse" is not being exercised. As written, the paper would be reporting
  a result on a *fixed, parameterization-imposed* near-one-step schedule.
- This also explains why the schedules at T=4 are *bitwise identical to four
  digits* across all six seeds and three block sizes; nothing data-adaptive
  is happening.

**Fix (pick one):** (i) reparameterize α with no saturating floor — e.g.
shift the offset from `−2` to something like `−6`, or use
`α = 0.5·sigmoid(cumsum(softplus(logits) − softplus(0)) − offset)`, or use
an unconstrained mapping that can drive α near 0 without saturating; verify
the schedule is no longer at a constraint after retraining. (ii) Freeze the
schedule to a chosen value (e.g. linear or cosine in α) and explicitly drop
the FLDD-style joint-learning framing — the paper becomes "block-factorized
discrete diffusion with a fixed schedule," which is still a perfectly
publishable result.

**B2. (was P0, now P1.)** See B2 in §C below — this is a wording issue, not
a methodology bug.

**B3. (was P0, now P1.)** See B3 in §C below — this is a comparison/framing
concern, not a methodology bug.

**B4. T=2 in E4 confounds reverse-head effect with forward-schedule effect.**
The README acknowledges this: at T=2, `|G|=4` reaches `α_T ≈ 0.50` while
`|G|=1` stays at `α_T ≈ 0.38`. The 42-point FID gap therefore mixes the
reverse-head expressiveness (the thing being tested) with the forward
schedule (a confound). As-is, the "block advantage grows as T shrinks"
claim is not cleanly attributable. **Fix:** rerun T=2 with the forward
schedule frozen to a common value across `|G| ∈ {1, 4}`.

**Important nuance on the choice of frozen α's.** The v1 chain (and our
initial `run_chain.sh`) used `FROZEN_ALPHAS="0.06 0.50"`. **These are
themselves the v1 parameterization-floor values from B1** — i.e. we'd be
"removing the confound" by freezing to the exact degenerate schedule that
B1 identifies as artefactual. That's internally inconsistent. A
methodologically clean choice is one of:

1. `α = [0.25, 0.50]` — linear interpolation between minimum useful
   corruption and uniform. Neutral, no v1 dependency.
2. The v2-learned `|G|=4` schedule at T=2 (run R1-T2 first, then freeze
   to that schedule for the |G|=1 control).
3. `α = [α_floor_v2, 0.50]` with `α_floor_v2 ≈ 0.001` — match the new
   parameterization floor; effectively a one-step schedule.

The headline claim survives if R3 with any of (1)–(3) keeps the block
advantage; the cleanest single experiment is (2).

**B6 (upgraded P1→P0). Best-checkpoint by training loss actually changes the
headline.** I previously listed this as P1 ("mildly favorable"). On re-check
the effect is concrete enough to upgrade. Across the three E2-from-ckpts
runs at |G|=1, the Pearson correlation between train-loss and FID is
**−1.000** — the *lowest* train-loss checkpoint has the *highest* FID, and
vice versa:

| seed | best epoch | train loss | FID |
|------|-----------|------------|-----|
| 42   | 89        | 689.53     | 58.58 |
| 43   | 97        | **691.10** | **55.78** |
| 44   | 81        | 690.27     | 57.21 |

For |G|=4 the correlation is +0.59 (positive but weak). With n=3 these are
unstable, but the direction is wrong for |G|=1 and means selecting by FID
(or by held-out NLL on a validation split) might give a lower FID for the
baseline, shrinking the headline gap. Cannot estimate by how much without
periodic checkpoints — please save every-K-epoch checkpoints and re-score.

**Fix:** (i) hold out 5–10% of MNIST train as a validation split, select
best.pt by val-ELBO; (ii) optionally also report final.pt as a no-selection
baseline; (iii) report FID-vs-epoch curves to show whether the win is
robust to checkpoint choice.

### P1 — Should address

**B2 (was P0). The "TV floor = 0.72" wording in `synthetic.py` is
misleading.** The number itself is correct as the TV from the data
distribution to *Uniform(16)* — i.e. the floor for an unconditional pixel-
factorized model that matches per-pixel marginals. But the docstring says
"A perfectly trained pixel-factorized model can therefore only represent
the uniform distribution over 16 states," which a reader will apply to the
pixel-factorized FLDD model in E1. That model is *conditional* on z_t via
the U-Net, and the marginal of z_0 across the chain is a mixture of
products — not constrained to uniform marginals over blocks. The empirical
|G|=1 TV of 0.31–0.43 (well below 0.72) confirms the unconditional floor is
not tight for FLDD. **Fix:** re-phrase as "TV of the best *z_t-blind*
independent-marginal predictor" or drop the reference entirely. The E1
result (block head reduces TV ~6.5×) stands either way.

**B3 (was P0). The unexplained gap to the prior 22.31 FID baseline is a
presentation problem, not a methodology bug.** PROBLEMSETTING cites the
previous DL-course TC-regularization result of FID 22.31 at T=4. The new
block-factorized headline is 49.08. Different architecture, training budget,
and possibly different binarization make these not directly comparable, but
a reader who only skims will conclude the new method is worse. **Fix
(pick one):** (i) enumerate the protocol differences in the paper text and
show they account for the gap (e.g. retrain the prior method under the new
protocol or vice versa); (ii) run the block head on top of the prior
method's backbone for a fair head-to-head; (iii) drop the 22.31 reference
if a direct comparison is out of scope, and frame the contribution as "vs.
own pixel-factorized baseline under matched protocol."

**B5. Cross-entropy ≠ ELBO when q is learned (now with numbers; gradient
direction CORRECTED from earlier v2 revision).** `compute_elbo_loss` uses
BCE / categorical CE between target `q(z_s|x)` and prediction
`p_θ(z_s|z_t)`. Cross-entropy = KL + H[q]. H[q] is independent of θ but
depends on the schedule φ via `target_pixel_prob`. Two consequences:

- **Loss magnitudes are misleading.** With the observed schedule
  (α ≈ [0.06, 0.06, 0.06, 0.50]), `H[Bern(0.06)] = 0.226 nats/pixel` and
  `H[Bern(0.5)] = 0.693 nats/pixel`. Per image:
  `parasitic = 784 · (3·0.226 + 0.693) ≈ 1075 nats summed across t=2..4`
  → after the `T·E_t[CE]` form: the parasitic contribution to the reported
  recon loss is **~531 nats** out of ~690. So roughly 77% of the reported
  "ELBO loss" is H[q], not KL. The numbers should not be compared to
  ELBO/NLL values in other discrete-diffusion papers.
- **The parasitic gradient on φ points toward *lower-entropy* α, not
  higher.** *(Corrected from the v2 revision, which had this inverted.)*
  Minimizing `CE = KL + H[target]` requires minimizing `H[target]`; for
  α ∈ (0, 0.5) the binary entropy `H[Bern(α)]` is monotone increasing in
  α, so `∂CE/∂α` from the H term is positive — driving α *downward*. This
  is the same direction the B1 parameterization floor was hard-stopping:
  in v1 the "early" α's pinned at 0.0596 precisely because both the data
  fit and the parasitic H gradient wanted lower α, and the floor caught
  both. After the joint B1 + B5 fix, early α's settle at ~0.024 — well
  above the new floor of 0.00124 — meaning the data does **not** in fact
  want α → 0 once the parasitic gradient is removed.

**Fix:** subtract `H[Bern(target_pixel_prob)]` from the per-pixel BCE
before summing (3-line change). The reported quantity is then a true ELBO
and the parasitic schedule gradient is gone.

**Empirical verification (v2 R1 + the independent reparam6_quick branch).**
The post-fix learned schedule converges to `[0.024, 0.25, 0.46, 0.494]`
across two completely different α parameterizations of the same fix —
this branch's `0.5·σ(cumsum(softplus(logits)) − 6)` and the parallel
branch's `0.5·(1 − exp(−cumsum(softplus(logits))))` form. Cross-replicated
to 2 decimal places. (i) confirms the fix has the intended effect on
schedule learning, (ii) confirms the v1 floor saturation was the joint
B1 + B5 effect rather than a property of the data, and (iii) supplies a
reviewer-grade cross-validation that the paper can cite directly.

**B7. Optional-stopping *risk* on E2 seeds (downgraded from "concern" to
"unknown, please confirm protocol").** The original sweep was seeds 42–44;
seeds 45–47 were added later via `results_e2_extra_bs{1,4}.json` /
`_extra_bs1_s45.json`. Adding the seeds *increased* the effect size: mean
Δ went from 6.04 (n=3) to 12.03 on the new seeds alone, 9.03 for n=6
combined. **I cannot prove optional-stopping from the data alone** — only
the protocol matters. But the data pattern is consistent with both
(a) genuine signal that grew with more seeds and (b) seed-fishing until
p<0.05. **Action needed:** the paper should state explicitly when the
n=3→n=6 decision was made and whether it was based on interim results.
If the answer is "we added more seeds after seeing the n=3 result," the
honest write-up is to either pre-register and rerun, or report the n=3
result as the headline with the n=6 as a follow-up replication.

**B8. Multiple comparisons not corrected.** Three pairwise tests (1v2, 1v4,
2v4) on the same six seeds. Holm-Bonferroni:

| pair | raw 1-sided p | Holm-adjusted |
|------|---------------|---------------|
| 2v4  | 0.001         | 0.003         |
| 1v4  | 0.006         | 0.012         |
| 1v2  | 0.105         | 0.105         |

Conclusion unchanged (2v4 and 1v4 remain significant), but the correction
should be reported.

**B9. FID protocol concerns.**

(i) `pytorch_fid` uses InceptionV3 features trained on ImageNet. On
single-channel hard-thresholded MNIST replicated to 3 channels, those
features are not well calibrated. Within-method rankings are robust, but
the absolute numbers don't directly compare to anything in the discrete
diffusion literature that uses an MNIST-classifier-FID. Either swap in a
MNIST-classifier feature extractor, or report a second metric (KID,
precision-recall) and footnote the choice.

(ii) FID at n=10k samples has Monte-Carlo noise of its own; the same E2
checkpoints re-scored in E4 give FIDs that drift by ~0.3 (e.g.
`bs1_s42_T=4`: 58.99 in E4, 58.58 in E2). Standard practice is n=50k, or
averaging 3+ sample sets per checkpoint. Either bump n or do the averaging,
and report sample-set variance separately from seed variance.

**B10. n=3 bootstrap CIs in E4 are degenerate.** README notes this — with
27 distinct resamples the 2.5/97.5 quantiles collapse to data min/max — but
the table still prints them as inferential intervals. Cleaner: drop the
bootstrap column for E4, keep only the paired t (df=2). Best fix: rerun
E4 at n=6 seeds to match E2 (you already have 6 at T=4).

### P2 — Nice-to-have

**B11. `|G|=2` only captures horizontal coupling.** A 1×2 block can absorb
horizontal within-block correlations but not vertical. The 1→2→4 monotone
story is partly explained by 2×2 capturing both axes that 1×2 misses. Adding
a 2×1 baseline at the same parameter count would disambiguate "axes
captured" from "block size."

**B12. Sampling-step argmax (downgraded from P1).** At the final reverse
step, `|G|=1` thresholds per pixel; `|G|=4` takes argmax over the 16-state
joint. On re-check this is each model operating under its native
factorization — not an unfair asymmetry. Worth noting in the paper but
not a defect. The cleanest write-up is "at t=1 each model takes the
argmax under its own factorization."

**B13. The TC decomposition is reported only one-sided.** E3 measures
within-block TC of the model. To complete the story, also report (i) the
within-block TC of the *data-averaged* target `E_{x|z_t}[q(z_s|x)]` (this
is what the model is fitting), and (ii) the *between*-block TC, which is
the irreducible residual the method explicitly does not address. Without
these you can't say how much of the gap the block head closes — only that
some closure happens.

**B14. E3 uses 3 seeds while E2 has 6.** The remaining bs4 checkpoints
(s45, s46, s47) already exist; rerun E3 against all six for parity.

**B15. T=4 row in E4 trained for 100 epochs (E2 reuse) while T∈{2,8,16}
trained for 80.** The README acknowledges this; cleanest is to retrain
T=4 at 80 epochs for the E4 table, or to footnote it on every row.

**B16. Standard deviation discrepancies.** The README E2 main table reports
"`|G|=1` ELBO 690.43 ± 0.71" and "FID 58.11 ± 2.96"; the JSON values give
sd = 0.81 and 2.88 respectively (numpy ddof=1, also matches `torch.std`
with default ddof). Likely a rounding / paste issue from an earlier version
of the numbers — easy to fix, but matters for a paper table.

**B17. `forward.py` stream-of-consciousness comments.** Lines 80–104 are an
inline "wait, let me reconsider" derivation that's helpful as a working
note but should be replaced by the clean final argument before public
release.

---

## C. What the paper can claim, and what it can't, as currently constituted

**Defensible claims:**
- On synthetic data with known within-block correlations, a 2×2-block head
  reduces block-state TV to ground truth from 0.31–0.43 to ~0.056 across
  three seeds (E1, H1 supported).
- On binarized MNIST at T=4, the 2×2-block head improves FID over the
  pixel-factorized baseline by ~9 FID points (paired Δ = 9.03, paired t p =
  0.006, six seeds; E2, H2 supported for |G|=4).
- The trained 2×2-block model exhibits substantially higher within-block TC
  on data-supported "mixed/stroke" regions than on "background" regions
  (E3, H3 supported).
- The block-vs-pixel FID gap is positive at every T ∈ {2,4,8,16} tested,
  with the gap markedly larger at T=2 (E4 directional).

**Cannot yet defend:**
- That the forward schedule is genuinely learned in any of these runs
  (B1) — it is saturated against the parameterization floor.
- That the T=2 widening of the block advantage is due to reverse-head
  expressiveness vs. forward-schedule choice (B4 confound).
- That the headline FID gap is robust to checkpoint selection criterion
  (B6 — best-by-train-loss is anti-correlated with FID at |G|=1, n=3).
- The "analytic 0.72 floor" framing for the pixel-factorized baseline (B2).
- Any comparison vs. the previous TC-regularization baseline at face value
  (B3 — protocol gap unaddressed).

Addressing B1, B4, B6 (true P0s) — and tightening B5/B7/B8/B9 — moves this
from "good DL-course report" to "publishable workshop / short paper"
caliber.

---

## D. Quick-checks performed

- Verified the 0.05960 parameterization floor of α analytically and against
  `schedule_summary.json` to four+ digits across all 18 E2 runs and all E4
  T=4/8/16 runs.
- Verified the 0.72 analytic TV from the synthetic distribution:
  `4·0.5·|0.2425−0.0625| + 12·0.5·|0.0025−0.0625| = 0.72` ✓
- Verified the n=6 paired t for FID 1v4:
  `t = 9.0327 / (5.6997/√6) = 3.88, p_one(df=5) = 0.0058` ✓ (matches README's
  0.006).
- Verified the 2×2 reshape correctness via explicit index expansion.
- Recomputed the per-row sd from the JSON; reported values are within ~2% of
  the JSON-derived values (see B16).

### Re-verifications (v2 pass)

- **B1 (parameterization floor) — CONFIRMED.** Floor `0.5·σ(−2) = 0.059601`.
  Max deviation of any "early" α from this floor across all 18 E2 ckpts is
  `5.2 × 10⁻⁵`. The schedule is saturated; no run learned away from the floor.
- **B2 (0.72 floor) — DOWNGRADED to P1.** Number is correct as the
  unconditional reference. Issue is wording in `synthetic.py` only.
- **B3 (gap to 22.31) — DOWNGRADED to P1.** Presentation issue, not a
  methodology bug.
- **B4 (T=2 confound) — CONFIRMED.** `T2_bs1` has α_T = 0.385 ± 0.004 across
  3 seeds; `T2_bs4` has α_T = 0.496 ± 0.003. The schedule difference is
  large and seed-stable.
- **B5 (BCE vs KL) — QUANTIFIED.** Parasitic H[q] term ~531 nats per image
  in the reported recon loss of ~690 — 77% of the value. Trivial 1-line
  fix.
- **B6 (best-by-train-loss) — UPGRADED to P0.** Across n=3 |G|=1 seeds,
  Pearson(train loss, FID) = −1.000. Best-by-train-loss does *not* pick
  the model with best FID; could meaningfully shift the headline.
- **B7 (optional stopping) — UNKNOWN, can't infer from data.** New seeds
  *helped* the hypothesis (mean Δ rose 6.04 → 9.03). Compatible with
  either "genuine signal grew with power" or "seed-fishing." Need
  protocol clarification.
- **B12 (argmax asymmetry) — DOWNGRADED to P2.** Each model uses argmax
  under its native factorization; this is correct behavior, not a bug.
- **B14 (E3 seed count) — CONFIRMED.** `results_e3.json` per_seed has only
  seeds 42/43/44, while bs4 checkpoints exist for 42–47. Free win to use
  all six.

---

## F. Next steps

Concrete, ordered plan. Estimated GPU hours assume the same scale as the
existing runs (single GPU, U-Net ~3M params, MNIST 100-epoch runs are
~30 min each on a recent GPU).

### F.1 — One-day rewrite (~0 GPU hours, just code + docs)

1. **Fix the strict-ELBO bug (B5).** In `compute_elbo_loss`, subtract
   `H[Bern(target_pixel_prob)]` from the per-pixel BCE for t > 1 (or
   equivalently compute KL directly). Re-derive and rename `recon_loss`
   → `recon_kl` so the table value is comparable to literature ELBO.
2. **Fix the misleading docstring (B2).** In `synthetic.py`,
   `pixel_factorized_tv_floor` → `unconditional_pixel_marginal_tv_floor`,
   docstring "best-possible TV for an unconditional pixel-factorized
   model" with an explicit note that the conditional FLDD model is not
   bound by this floor.
3. **Add multiple-comparison correction (B8).** One paragraph in
   `merge_e2_stats.py` output + README table.
4. **Drop bootstrap CIs from E4 tables (B10),** keep only paired-t.
5. **Clean the stream-of-consciousness comments in `forward.py` lines
   80–104** (B17).
6. **Footnote the |G|=2 horizontal-only caveat (B11)** in the E2 narrative
   and the FID-protocol caveats (B9) — InceptionV3-on-MNIST, n=10k.

### F.2 — Critical re-runs (~12 GPU hours)

7. **Reparameterize the schedule (B1).** Change the offset from −2 to −6
   (or use `softplus(logits) − softplus(0)` so per-step α can hit 0). Re-run
   all 18 E2 configurations. Two outcomes:
   - **If schedules now diverge across runs / block sizes** → the FLDD
     framing is rescued; the schedule comparison story is now real and
     paper-worthy.
   - **If schedules still saturate at the new (lower) floor** → conclude
     "the model wants α ≈ 0 in the no-op steps" and recast the paper as
     "block-factorized discrete diffusion with a near-degenerate schedule
     on binarized MNIST." Either is publishable; neither is the current
     framing.
8. **Held-out validation split for checkpoint selection (B6).** Hold out
   5–10% of the MNIST train set, use val-ELBO to pick best.pt. Re-score
   all 12 E2 ckpts (6 seeds × 2 block sizes). If the headline Δ FID
   changes by >2, that becomes the new headline number.
9. **T=2 schedule-controlled rerun (B4).** Re-run E4's T=2 row with the
   forward schedule frozen to a common α for both block sizes. **Use the
   v2-learned |G|=4 T=2 schedule, NOT [0.06, 0.5]** — the latter is the
   v1 floor and re-introduces the artefact B1 is supposed to remove.
   Disambiguates the T=2 confound.

### F.3 — Statistical-power top-ups (~10 GPU hours)

10. **Bump E3 to all 6 |G|=4 seeds (B14).** Free, the ckpts already exist.
11. **Either re-run E2 with the n=6 set pre-registered (B7), or report
    n=3 alongside n=6** with explicit narrative on which decision came first.
12. **Bump E4 to n=6 seeds at T ∈ {2, 8, 16}** to match E2. ~6 GPU hours.
13. **FID Monte-Carlo noise (B9.ii).** Generate 3 independent sample sets
    per checkpoint, report FID-of-each + sample-set-sd separately from
    seed-sd. ~10× cheaper than retraining.

### F.4 — Strengthening the science (optional but high-leverage, ~20–40 GPU hours)

14. **Add a 2×1 (vertical) |G|=2 baseline (B11).** Disambiguates
    "axes captured" from "block size" — the cleanest version of the
    monotonicity story.
15. **Quantify the within- vs between-block TC of the *target* (B13).**
    For a given trained checkpoint, sample (x, z_t) pairs and estimate
    `TC[E_{x|z_t}[q(z_s|x)]]` decomposed into within-block + between-block
    components. This closes the theoretical loop: "we absorbed X nats out
    of Y total within-block TC, leaving Z nats of irreducible between-block
    residual." Without this, the paper has only an FID-shaped story.
16. **Head-to-head against the prior 22.31 baseline (B3).** Pick one:
    (a) run the block head on top of the prior TC-regularization codebase;
    (b) run the prior TC-regularization on the new codebase; (c) drop the
    comparison entirely and reposition as "vs. pixel-factorized baseline
    under matched protocol." Pick (c) if compute is tight; (a) is the
    strongest scientifically.

### F.5 — Paper-writing checklist (after F.1–F.3)

- One-line story per experiment, then numbers, then caveats — in that order.
- For every claim, name the test, the n, the sided-ness, and the correction.
- For every plot, the underlying numerical table in an appendix.
- The "schedule collapse" plot should be re-rendered after F.7 to show the
  *new* (post-reparameterization) schedule alongside the old one.
- Clearly state in §2 (method) that all training uses the same forward-
  process module across block sizes, and that B4 was specifically addressed
  for T=2 in §5 (E4).

### F.6 — Recommended go/no-go before submission

Submit when **F.1 + F.2 are done** and the headline still holds (Δ FID at
T=4, |G|=1 vs |G|=4, p<0.05 after Holm correction, n≥6, schedule-fix
verified, best-by-val-ELBO selection). If F.2.7 reveals the schedule was
the real story (or wasn't), the paper repositions but stays
publishable. If F.2.8 shrinks Δ FID below noise, hold the paper and
rerun with more compute / longer training.

### F.7 — v3 follow-up (after the v2 audit-of-audit)

Two priority actions before more seed sweeps:

17. **Add EMA (B18).** ~30 lines in `fldd/train.py`. Standard recipe:
    decay 0.9999, warmup `(1+step)/(10+step)`, EMA updated after every
    optimizer step, EMA-weighted forward used for validation and FID
    scoring. See the v3 `fldd/train.py` / `train_mnist.py` for the
    reference implementation.
18. **Four-cell pilot ablation** before any large rerun. bs4 seed 42
    only, 50 epochs, all with `loss_form=true_elbo`,
    `sigmoid_offset=-6`:
    | EMA \\ select | train_loss | val_loss |
    |---|---|---|
    | off | (a) v2 minus B6 | (b) current v2 baseline |
    | on  | (c) v2 + EMA, train sel | (d) v2 + EMA, val sel |
    Then a single bs1 seed 42 run at the winning config to confirm the
    block advantage isn't destroyed by EMA. ~2.5 GPU hours total.
    Whichever cell minimizes FID dictates the headline rerun config.
19. **B4 frozen-α rerun: use the v2 |G|=4 T=2 learned schedule, not
    `[0.06, 0.50]`.** Re-running R3 with the v1-floor values reintroduces
    the artefact B1 is supposed to remove. Block this step on the R1-T2
    schedule landing.
