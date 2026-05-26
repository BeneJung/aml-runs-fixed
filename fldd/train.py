import torch
import torch.nn.functional as F

from fldd.blocks import compute_block_target


def _bernoulli_entropy(p, eps=1e-7):
    """Per-element H[Bern(p)] in nats."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _categorical_entropy(probs, eps=1e-7, dim=1):
    """H[Categorical(probs)] in nats, summed over `dim`."""
    p = probs.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=dim)


def compute_elbo_loss(
    model,
    forward_process,
    x,
    T,
    block_size=1,
    loss_form="true_elbo",
    orientation="horizontal",
):
    """Compute the discrete diffusion training loss.

    Supports both pixel-factorized (block_size=1) and block-factorized
    (block_size in {2, 4}) reverse heads.

    The variational bound decomposes as

        L = sum_{t=1}^{T} E_{z_t}[ KL[q(z_{t-1}|z_t,x) || p_theta(z_{t-1}|z_t)] ]
            + KL[q(z_T|x) || p(z_T)]

    with the t=1 KL collapsing to -log p_theta(x|z_1) because
    q(z_0|z_1,x) = delta(x). For t > 1 we use the FLDD identity
    q(z_s|z_t,x) = q(z_s|x) (non-Markovian forward, see forward.py).

    `loss_form` controls how the t > 1 term is computed:

    - "true_elbo" (default; B5 fix):
          KL[ q(z_s|x) || p_theta(z_s|z_t) ]
      which equals CE - H[q(z_s|x)]. This is what the ELBO bound
      actually contains, and what should be reported as "ELBO loss"
      in any table compared against the literature.

    - "ce" (original codebase behaviour; kept for reproducibility of
       the old E2/E4 numbers):
          CE[ q(z_s|x), p_theta(z_s|z_t) ] = KL + H[q]
      With learned alphas, the H[q] term has gradient w.r.t. the schedule,
      which leaks an entropy-maximising bias into the schedule update.
      Quantitatively, with the historical alpha schedule on binarized
      MNIST T=4, ~77% of the reported recon loss is H[q], not KL.

    Both forms have the same gradient w.r.t. the *reverse model* theta,
    so they train identically when the schedule is held fixed.
    """
    if loss_form not in ("true_elbo", "ce"):
        raise ValueError(f"loss_form must be 'true_elbo' or 'ce', got {loss_form!r}")

    device = x.device
    B = x.shape[0]

    # sample a random timestep t uniformly from {1, ..., T}
    t = torch.randint(1, T + 1, (B,), device=device)

    # sample z_t ~ q(z_t | x)
    alphas = forward_process.get_alphas()
    alpha_t = alphas[t - 1]  # (B,)
    prob_one_zt = (
        x * (1.0 - alpha_t[:, None, None, None])
        + (1.0 - x) * alpha_t[:, None, None, None]
    )
    z_t = torch.bernoulli(prob_one_zt)

    # model prediction
    logits = model(z_t, t - 1)  # 0-indexed timestep

    # per-pixel target probabilities for z_{t-1}
    is_first = (t == 1).float()[:, None, None, None]
    alpha_s = alphas[torch.clamp(t - 2, min=0)]
    target_pixel_prob = (
        x * (1.0 - alpha_s[:, None, None, None])
        + (1.0 - x) * alpha_s[:, None, None, None]
    )
    # for t = 1, target collapses to delta(x)
    target_pixel_prob = is_first * x + (1.0 - is_first) * target_pixel_prob

    if block_size == 1:
        pred_prob = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        # CE term per pixel
        ce = -(
            target_pixel_prob * torch.log(pred_prob)
            + (1.0 - target_pixel_prob) * torch.log(1.0 - pred_prob)
        )  # (B, 1, H, W)
        per_pixel = ce
        if loss_form == "true_elbo":
            # Subtract H[Bern(target)] but only for t > 1; for t = 1 the
            # target is delta(x) and H = 0 (CE already equals -log p(x|z_1)).
            H_target = _bernoulli_entropy(target_pixel_prob)
            per_pixel = per_pixel - (1.0 - is_first) * H_target
        reconstruction_loss = T * per_pixel.sum(dim=(1, 2, 3)).mean()
    else:
        # block-factorized: cross-entropy / KL over block categorical
        target_dist = compute_block_target(
            target_pixel_prob, block_size, orientation=orientation,
        )  # (B, K^|G|, Hb, Wb)
        log_pred = F.log_softmax(logits, dim=1)
        ce = -(target_dist * log_pred).sum(dim=1)  # (B, Hb, Wb)
        per_block = ce
        if loss_form == "true_elbo":
            # H[Categorical(target_dist)] over the K^|G| states. For t = 1
            # the target is one-hot (collapsed delta), H = 0; we mask
            # below to be explicit and numerically safe.
            H_target = _categorical_entropy(target_dist, dim=1)  # (B, Hb, Wb)
            per_block = per_block - (1.0 - is_first.squeeze(1)) * H_target
        reconstruction_loss = T * per_block.sum(dim=(1, 2)).mean()

    # prior loss: KL[q(z_T|x) || Uniform]
    prior_loss = forward_process.kl_prior(x)

    loss = reconstruction_loss + prior_loss

    metrics = {
        "loss": loss.item(),
        "recon": reconstruction_loss.item(),
        "prior": prior_loss.item(),
        "loss_form": loss_form,
    }
    return loss, metrics


@torch.no_grad()
def compute_validation_elbo(
    model,
    forward_process,
    val_loader,
    T,
    device,
    block_size=1,
    loss_form="true_elbo",
    samples_per_t=4,
    orientation="horizontal",
):
    """Estimate the held-out ELBO with reduced Monte Carlo noise.

    Unlike compute_elbo_loss (which samples ONE t per image), here we
    iterate over all t in {1, ..., T} and average `samples_per_t` z_t
    samples per t. With T = 4 and samples_per_t = 4 this is 16x more
    stable than the training-time stochastic estimate, which matters
    when this quantity is used to pick best.pt (B6).

    Returns a dict with mean per-image recon, prior, and total loss.
    """
    model.eval()
    forward_process.eval()

    total_recon = 0.0
    total_prior = 0.0
    n_images = 0

    for (x,) in val_loader:
        x = x.to(device)
        B = x.shape[0]

        per_image_recon = torch.zeros(B, device=device)
        alphas = forward_process.get_alphas()

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
                z_t = torch.bernoulli(prob_one_zt)
                logits = model(z_t, t - 1)

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
                    accum = accum + per.sum(dim=(1, 2, 3))
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
                    accum = accum + per.sum(dim=(1, 2))

            per_image_recon = per_image_recon + accum / samples_per_t

        # NOTE: we are computing sum_t E_{z_t}[KL_t] directly — no T multiplier.
        # This is the deterministic sum, not the random-t MC estimate.
        prior_per_image = forward_process.kl_prior(x).item() * B
        total_recon += per_image_recon.sum().item()
        total_prior += prior_per_image
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
):
    model.train()
    forward_process.train()
    total_metrics = {"loss": 0.0, "recon": 0.0, "prior": 0.0}
    n_batches = 0

    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, metrics = compute_elbo_loss(
            model, forward_process, x, T, block_size, loss_form=loss_form,
            orientation=orientation,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(forward_process.parameters()), 1.0
        )
        optimizer.step()

        total_metrics["loss"] += metrics["loss"]
        total_metrics["recon"] += metrics["recon"]
        total_metrics["prior"] += metrics["prior"]
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}
