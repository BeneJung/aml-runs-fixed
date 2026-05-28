"""Training loss, EMA, validation, and one-epoch driver.

This is the v2-audit-revised version. Differences from the pre-audit repo:

1.  **`loss_form` defaults to "true_elbo"** (B5 fix; the previous "ce" default
    is still selectable to reproduce the historical loss values exactly).
2.  **`EMA` class added** and a fast in-place context manager (`use_ema`).
    Diffusion FID is notoriously sensitive to EMA-smoothed weights; the
    pre-audit pipeline had none. Decay 0.9999 with a `(1+step)/(10+step)`
    warmup is the standard recipe.
3.  **`compute_validation_elbo` takes a `Generator` argument** so the val
    ELBO is bit-reproducible across calls (ported from teammate's
    `evaluate_elbo`). `samples_per_t=1` is the cheap default now that the
    estimator is deterministic per (epoch, seed).
4.  **`train_epoch` accepts and updates the EMA each step.**

Backwards compatibility: pass `ema=None` and `loss_form="ce"` to recover
the original pre-audit behaviour exactly.
"""

import copy
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from fldd.blocks import compute_block_target


# ----------------------------------------------------------------------------
# Helpers: per-element entropies used by the true-ELBO subtraction
# ----------------------------------------------------------------------------

def _bernoulli_entropy(p, eps=1e-7):
    """Per-element H[Bern(p)] in nats."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _categorical_entropy(probs, eps=1e-7, dim=1):
    """H[Categorical(probs)] in nats, summed over `dim`."""
    p = probs.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=dim)


# ----------------------------------------------------------------------------
# EMA — exponential moving average of model parameters
# ----------------------------------------------------------------------------

class EMA:
    """Shadow copy of a model's trainable parameters, updated each optimizer step.

    `update(model)` should be called *after* `optimizer.step()`. The effective
    decay is `min(self.decay, (1+step)/(10+step))` if `use_warmup`, otherwise
    `self.decay`. With decay=0.9999 the warmup is ~negligible after ~100 steps.

    For inference, use the `use_ema` context manager below — it swaps the EMA
    weights into the model in-place and restores the live weights on exit.
    """

    def __init__(self, model, decay=0.9999, use_warmup=True):
        self.decay = float(decay)
        self.use_warmup = bool(use_warmup)
        self.step = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def _current_decay(self):
        if not self.use_warmup:
            return self.decay
        warm = (1.0 + self.step) / (10.0 + self.step)
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model):
        d = self._current_decay()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)
        self.step += 1

    @torch.no_grad()
    def copy_to(self, model):
        """Overwrite `model`'s parameters with the EMA shadow weights, in-place."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {
            "decay": self.decay,
            "use_warmup": self.use_warmup,
            "step": self.step,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd, device=None):
        self.decay = float(sd["decay"])
        self.use_warmup = bool(sd.get("use_warmup", True))
        self.step = int(sd["step"])
        self.shadow = {
            k: (v.to(device) if device is not None else v).clone()
            for k, v in sd["shadow"].items()
        }


@contextmanager
def use_ema(model, ema):
    """Context manager: swap EMA shadow weights into `model` for the duration.

    If `ema` is None, this is a noop. Live weights are restored on exit even
    if the wrapped code raises.
    """
    if ema is None:
        yield
        return
    backup = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in ema.shadow
    }
    try:
        ema.copy_to(model)
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])


# ----------------------------------------------------------------------------
# Training loss
# ----------------------------------------------------------------------------

def compute_elbo_loss(
    model,
    forward_process,
    x,
    T,
    block_size=1,
    loss_form="true_elbo",
    orientation="horizontal",
):
    """Discrete-diffusion training loss.

    `loss_form`:
      * "true_elbo" (default; B5 fix) — `KL[q(z_s|x) || p_theta(z_s|z_t)]`,
        the actual ELBO contribution.
      * "ce" — `CE[q(z_s|x), p_theta(z_s|z_t)] = KL + H[q]`. Reproduces the
        pre-audit reported loss values. Minimizing CE places a bias on the
        schedule φ in the direction of *lower* H[target], i.e. toward
        *smaller* α (more concentrated targets) — see REVIEW.md §B5. That's
        why every early α saturated against the parameterization floor in
        the v1 runs.

    Both forms share the gradient w.r.t. the reverse-model θ; they differ
    only in the φ gradient and in the absolute loss value.
    """
    if loss_form not in ("true_elbo", "ce"):
        raise ValueError(f"loss_form must be 'true_elbo' or 'ce', got {loss_form!r}")

    device = x.device
    B = x.shape[0]

    t = torch.randint(1, T + 1, (B,), device=device)
    alphas = forward_process.get_alphas()
    alpha_t = alphas[t - 1]
    prob_one_zt = (
        x * (1.0 - alpha_t[:, None, None, None])
        + (1.0 - x) * alpha_t[:, None, None, None]
    )
    z_t = torch.bernoulli(prob_one_zt)
    logits = model(z_t, t - 1)

    is_first = (t == 1).float()[:, None, None, None]
    alpha_s = alphas[torch.clamp(t - 2, min=0)]
    target_pixel_prob = (
        x * (1.0 - alpha_s[:, None, None, None])
        + (1.0 - x) * alpha_s[:, None, None, None]
    )
    target_pixel_prob = is_first * x + (1.0 - is_first) * target_pixel_prob

    if block_size == 1:
        pred_prob = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        ce = -(
            target_pixel_prob * torch.log(pred_prob)
            + (1.0 - target_pixel_prob) * torch.log(1.0 - pred_prob)
        )
        per = ce
        if loss_form == "true_elbo":
            H_target = _bernoulli_entropy(target_pixel_prob)
            per = per - (1.0 - is_first) * H_target
        reconstruction_loss = T * per.sum(dim=(1, 2, 3)).mean()
    else:
        target_dist = compute_block_target(
            target_pixel_prob, block_size, orientation=orientation,
        )
        log_pred = F.log_softmax(logits, dim=1)
        ce = -(target_dist * log_pred).sum(dim=1)
        per = ce
        if loss_form == "true_elbo":
            H_target = _categorical_entropy(target_dist, dim=1)
            per = per - (1.0 - is_first.squeeze(1)) * H_target
        reconstruction_loss = T * per.sum(dim=(1, 2)).mean()

    prior_loss = forward_process.kl_prior(x)
    loss = reconstruction_loss + prior_loss
    metrics = {
        "loss": loss.item(),
        "recon": reconstruction_loss.item(),
        "prior": prior_loss.item(),
        "loss_form": loss_form,
    }
    return loss, metrics


# ----------------------------------------------------------------------------
# Validation ELBO (deterministic, seeded)
# ----------------------------------------------------------------------------

@torch.no_grad()
def compute_validation_elbo(
    model,
    forward_process,
    val_loader,
    T,
    device,
    block_size=1,
    loss_form="true_elbo",
    samples_per_t=1,
    orientation="horizontal",
    seed=0,
):
    """Deterministic full-T held-out ELBO.

    Iterates over all t ∈ {1,...,T} and averages `samples_per_t` z_t draws
    per (image, t). A fresh `torch.Generator` is seeded with `seed` at the
    start, so the returned number is **bit-reproducible across epochs and
    runs** given the same model + val_loader (ported from teammate's
    `evaluate_elbo`). `samples_per_t=1` is enough for a held-out set of
    several thousand images: per-image variance is averaged out across the
    set, and reproducibility matters more than per-image variance for
    picking the best checkpoint.
    """
    model.eval()
    forward_process.eval()

    gen = torch.Generator(device=device).manual_seed(int(seed))

    total_recon = 0.0
    total_prior = 0.0
    n_images = 0

    alphas = forward_process.get_alphas()

    for (x,) in val_loader:
        x = x.to(device)
        B = x.shape[0]

        per_image_recon = torch.zeros(B, device=device)

        for t_int in range(1, T + 1):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            alpha_t = alphas[t - 1]
            is_first = (t == 1).float()[:, None, None, None]
            alpha_s = alphas[torch.clamp(t - 2, min=0)]
            target_pixel_prob = (
                x * (1.0 - alpha_s[:, None, None, None])
                + (1.0 - x) * alpha_s[:, None, None, None]
            )
            target_pixel_prob = is_first * x + (1.0 - is_first) * target_pixel_prob

            accum = torch.zeros(B, device=device)
            for _ in range(samples_per_t):
                prob_one_zt = (
                    x * (1.0 - alpha_t[:, None, None, None])
                    + (1.0 - x) * alpha_t[:, None, None, None]
                )
                z_t = torch.bernoulli(prob_one_zt, generator=gen)
                logits = model(z_t, t - 1)

                if block_size == 1:
                    pred_prob = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
                    ce = -(
                        target_pixel_prob * torch.log(pred_prob)
                        + (1.0 - target_pixel_prob) * torch.log(1.0 - pred_prob)
                    )
                    per = ce
                    if loss_form == "true_elbo":
                        per = per - (1.0 - is_first) * _bernoulli_entropy(target_pixel_prob)
                    accum = accum + per.sum(dim=(1, 2, 3))
                else:
                    target_dist = compute_block_target(
                        target_pixel_prob, block_size, orientation=orientation,
                    )
                    log_pred = F.log_softmax(logits, dim=1)
                    ce = -(target_dist * log_pred).sum(dim=1)
                    per = ce
                    if loss_form == "true_elbo":
                        per = per - (1.0 - is_first.squeeze(1)) * _categorical_entropy(target_dist, dim=1)
                    accum = accum + per.sum(dim=(1, 2))

            per_image_recon = per_image_recon + accum / samples_per_t

        prior_per_image = forward_process.kl_prior(x).item()
        total_recon += per_image_recon.sum().item()
        total_prior += prior_per_image * B
        n_images += B

    model.train()
    forward_process.train()

    mean_recon = total_recon / n_images
    mean_prior = total_prior / n_images
    return {
        "val_recon": mean_recon,
        "val_prior": mean_prior,
        "val_loss": mean_recon + mean_prior,
    }


# ----------------------------------------------------------------------------
# One epoch
# ----------------------------------------------------------------------------

def train_epoch(
    model,
    forward_process,
    train_loader,
    optimizer,
    T,
    device,
    block_size=1,
    loss_form="true_elbo",
    orientation="horizontal",
    ema=None,
    grad_clip=1.0,
):
    """Single training epoch.

    If `ema` is an `EMA` instance, its shadow weights are updated after
    every optimizer step.
    """
    model.train()
    forward_process.train()
    total_metrics = {"loss": 0.0, "recon": 0.0, "prior": 0.0}
    n_batches = 0

    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, metrics = compute_elbo_loss(
            model, forward_process, x, T, block_size,
            loss_form=loss_form, orientation=orientation,
        )
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(forward_process.parameters()),
                grad_clip,
            )
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_metrics["loss"] += metrics["loss"]
        total_metrics["recon"] += metrics["recon"]
        total_metrics["prior"] += metrics["prior"]
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}
