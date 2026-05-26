# Closing the Factorization Gap in Discrete Diffusion via Locally-Coupled Reverse Models

**AML 2026 Semester Project — Proposal**

Kai Koepchen (24-738-189), Benedikt Jung (24-723-082), Ardjan Doci (17-878-349), Peng Liu (23-742-455) — University of Zurich

## Background

Discrete diffusion models generate data **x** in {1,...,K}^D by learning to reverse a corruption process. Forward-Learned Discrete Diffusion (FLDD) learns a *data-aware* noise schedule jointly with the reverse model, enabling generation in as few as T = 4 steps.

However, this data-awareness introduces a **factorization gap**: the true reverse target q(z_s | z_t) — averaged over unknown data **x** — develops cross-pixel correlations, but the standard reverse model p_theta predicts each pixel independently. This mismatch is characterized by the **Total Correlation (TC)** of the aggregated posterior, and it is irreducible regardless of the reverse model's capacity.

### Limitations of prior approaches

A prior DL course project attempted to reduce TC by penalizing q_phi during training (L = ELBO + lambda * TC), alongside entropic OT coupling and a REINFORCE baseline. This yielded FID 40.80 -> 22.31 on MNIST (T=4), but the approach has structural limitations:
- The TC estimator is high-variance at D=784, making the regularization signal noisy
- Penalizing TC pushes q_phi toward a data-blind schedule, conflicting with FLDD's core benefit
- Extreme sensitivity to lambda (FID 22.31 at 10^-3 vs 52.57 at 10^-2)

## Our Approach: Block-Factorized Reverse Models

**Key insight:** Instead of reducing TC in the forward process (which degrades the schedule), we *absorb* local TC in the reverse model by predicting joint distributions over small blocks of adjacent pixels.

Given a block partition G = {G_1, ..., G_B}:

```
p_theta(z_s | z_t) = prod_b p_theta(z_s^{G_b} | z_t)
```

where each block output is a categorical distribution over K^|G_b| joint states. The KL loss decomposes as:

```
KL[q || p_theta] = TC_between (irreducible) + sum_b KL[q^{G_b} || p_theta^{G_b}] (reducible)
```

Block-factorized p_theta absorbs within-block TC by construction, reducing the irreducible gap from TC_total to TC_between.

For binarized MNIST (K=2), 2x2 blocks yield a 16-dim softmax per block — smaller than a typical classification head and computationally cheap.

**How TC absorption works:** Each individual training target q_phi(z_s^G | z_t, x) is factorized (coupling is element-wise). But different data points x produce different factorized targets for the same z_t. Over training, the block model learns the data-averaged target — a mixture of product distributions — which is a joint distribution capturing within-block correlations. A pixel-factorized model is forced to collapse this to a product of marginals.

## Problem Setting

**Goal:** Investigate whether block-factorized p_theta can absorb local TC and improve sample quality over the standard pixel-factorized p_theta in FLDD at T=4 on binarized MNIST.

We scope this project to a single dataset (binarized MNIST, K=2) and a single architecture modification (block output head). A positive result shows local correlations matter; a negative result shows correlations are predominantly long-range. Both are informative.

## Experiments

- **E1 — Synthetic validation:** Construct a small synthetic binary-image dataset with known local correlations. Verify that block-factorized models outperform pixel-factorized models. Validates the theory in a controlled setting.
- **E2 — Block size vs. FID on MNIST:** Fix T=4. Train FLDD with |G| in {1, 2, 4}. Report FID over >= 3 seeds. Core result: does block-factorization improve sample quality?
- **E3 — Block joint analysis:** Visualize learned block-level distributions for stroke vs. background regions. Measure deviation from the factorized form as evidence of TC absorption.
- **E4 (stretch goal) — Steps vs. quality:** Fix best |G| from E2. Plot FID vs. T in {2, 4, 8, 16}. Does the block advantage grow at lower T?

## Hypotheses

- **H1:** On the synthetic dataset with known local correlations, block-factorized p_theta (|G|=4) achieves better sample quality than pixel-factorized p_theta (|G|=1).
- **H2:** On MNIST at T=4, block-factorized p_theta achieves lower FID than pixel-factorized p_theta, with the same forward process.
- **H3:** Block-level joints in stroke regions exhibit stronger deviation from the factorized form than background regions, providing direct evidence of local TC absorption.

If H2 is rejected, this implies cross-pixel correlations in FLDD are predominantly long-range rather than local — an informative finding that would motivate autoregressive or attention-based reverse models.

## Deliverables

1. **Implementation** of block-factorized output head for the existing FLDD codebase, supporting |G| in {1, 2, 4}.
2. **Synthetic experiment** (E1) demonstrating block factorization on a controlled dataset.
3. **MNIST experiment** (E2) comparing FID of pixel- vs. block-factorized models at T=4 over >= 3 seeds.
4. **Qualitative analysis** (E3) of learned block joints in stroke vs. background regions.
5. **Final report** presenting results, including interpretation of both positive and negative findings.

The stretch goal (E4) will be pursued if time permits but is not part of our core commitment.

## Relation to Prior Work

This project builds on a DL course project that identified the TC gap in FLDD and attempted to address it via TC regularization and OT coupling. Our block-factorized approach offers a complementary strategy: instead of modifying the forward process, we increase the expressiveness of the reverse model to absorb local correlations directly.
