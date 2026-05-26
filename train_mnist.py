"""Training entry point for binarized MNIST.

Now supports (B1) configurable forward-schedule sigmoid offset,
(B4) fixed-schedule training, (B5) true-ELBO loss form, and
(B6) held-out-validation checkpoint selection. All defaults preserve
the original behaviour so existing experiments remain reproducible;
opt into the fixes via the matching CLI flags.
"""

import argparse
import os
import torch
from tqdm import tqdm

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.train import train_epoch, compute_validation_elbo
from fldd.sample import sample, save_samples


def run_mnist(
    block_size=1,
    seed=42,
    T=4,
    epochs=100,
    batch_size=128,
    lr=3e-4,
    device="cuda",
    save_dir="checkpoints",
    save_ckpt_as_best="best.pt",
    save_ckpt_as_final="final.pt",
    sample_every=10,
    samples_dir="samples",
    verbose=True,
    # --- new options ---
    sigmoid_offset=LearnedForwardProcess.HISTORICAL_OFFSET,  # B1
    fixed_alphas=None,                                       # B4
    loss_form="ce",                                          # B5
    val_fraction=0.0,                                        # B6
    select_by="train_loss",                                  # B6
    val_samples_per_t=4,                                     # B6
    split_seed=0,                                            # B6
    orientation="horizontal",                                # B11
):
    """Train FLDD on binarized MNIST.

    New options (all default to original behaviour for backward compat):
        sigmoid_offset (B1): forward-process parameterization offset.
            Default -2.0 (historical) gives alpha floor ~0.0596 which the
            optimizer saturates against in every E2 run. Try -6.0 for a
            floor of ~0.00124 if you want the schedule to actually learn.
        fixed_alphas (B4): if set, freezes the schedule to these alphas.
            Use to control-out the schedule when comparing block sizes
            (the T=2 confound in E4).
        loss_form (B5): "ce" (original) or "true_elbo" (strict KL).
            Defaults to "ce" so reported losses match older runs.
        val_fraction (B6): if > 0, hold out this fraction of train as
            validation. Default 0 = no val split.
        select_by (B6): "train_loss" (original) or "val_loss".
            "val_loss" requires val_fraction > 0.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    if verbose:
        print(
            f"Training FLDD on binarized MNIST | T={T} block_size={block_size} "
            f"seed={seed} device={device}"
        )
        print(
            f"  sigmoid_offset={sigmoid_offset}  "
            f"fixed_alphas={'set' if fixed_alphas is not None else 'no'}  "
            f"loss_form={loss_form}  "
            f"select_by={select_by}  val_fraction={val_fraction}"
        )

    use_val = val_fraction > 0.0
    if select_by == "val_loss" and not use_val:
        raise ValueError("select_by='val_loss' requires val_fraction > 0")

    if use_val:
        train_loader, val_loader, _ = get_binarized_mnist(
            batch_size=batch_size,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
    else:
        train_loader, _ = get_binarized_mnist(batch_size=batch_size)
        val_loader = None

    forward_process = LearnedForwardProcess(
        T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas,
    ).to(device)
    if orientation == "vertical" and block_size != 2:
        raise ValueError(
            f"orientation='vertical' only applies to block_size=2; "
            f"got block_size={block_size}"
        )
    model = UNet(
        channels=(32, 64, 128), block_size=block_size,
        orientation=orientation,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  U-Net parameters: {n_params:,}")
        print(f"  alpha floor (parameterization): "
              f"{forward_process.alpha_floor:.6f}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forward_process.parameters()),
        lr=lr,
    )

    os.makedirs(save_dir, exist_ok=True)
    if sample_every and samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    best_score = float("inf")
    best_epoch = None
    metrics = None
    val_history = []

    for epoch in tqdm(range(1, epochs + 1), desc="training", disable=not verbose):
        metrics = train_epoch(
            model, forward_process, train_loader, optimizer,
            T, device, block_size, loss_form=loss_form,
            orientation=orientation,
        )

        val_metrics = None
        if use_val:
            val_metrics = compute_validation_elbo(
                model, forward_process, val_loader, T, device,
                block_size=block_size, loss_form=loss_form,
                samples_per_t=val_samples_per_t,
                orientation=orientation,
            )
            val_history.append({"epoch": epoch, **val_metrics})

        alphas = forward_process.get_alphas().detach().cpu().tolist()
        alpha_str = ", ".join(f"{a:.4f}" for a in alphas)
        if verbose:
            line = (
                f"epoch {epoch:3d} | loss {metrics['loss']:.4f} | "
                f"recon {metrics['recon']:.4f} | prior {metrics['prior']:.4f}"
            )
            if val_metrics is not None:
                line += (
                    f" | val_loss {val_metrics['val_loss']:.4f}"
                    f" (recon {val_metrics['val_recon']:.4f})"
                )
            line += f" | alphas [{alpha_str}]"
            print(line)

        # checkpoint selection
        if save_ckpt_as_best is not None:
            score = (val_metrics["val_loss"] if select_by == "val_loss"
                     else metrics["loss"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "forward": forward_process.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": metrics["loss"],
                    "val_loss": val_metrics["val_loss"] if val_metrics else None,
                    "select_by": select_by,
                    "block_size": block_size,
                    "T": T,
                    "seed": seed,
                    "sigmoid_offset": sigmoid_offset,
                    "fixed_alphas": (
                        fixed_alphas.tolist() if hasattr(fixed_alphas, "tolist")
                        else list(fixed_alphas) if fixed_alphas is not None
                        else None
                    ),
                    "loss_form": loss_form,
                }, os.path.join(save_dir, save_ckpt_as_best))

        if (sample_every and sample_every > 0 and samples_dir
                and epoch % sample_every == 0):
            samples = sample(model, forward_process, T, n_samples=64,
                             device=device, block_size=block_size,
                             orientation=orientation)
            save_samples(samples, os.path.join(samples_dir, f"epoch_{epoch:03d}.png"))
            if verbose:
                print(f"  -> saved samples to {samples_dir}/epoch_{epoch:03d}.png")

    if save_ckpt_as_final is not None:
        torch.save({
            "epoch": epochs,
            "model": model.state_dict(),
            "forward": forward_process.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": metrics["loss"],
            "block_size": block_size,
            "T": T,
            "seed": seed,
            "sigmoid_offset": sigmoid_offset,
            "fixed_alphas": (
                fixed_alphas.tolist() if hasattr(fixed_alphas, "tolist")
                else list(fixed_alphas) if fixed_alphas is not None
                else None
            ),
            "loss_form": loss_form,
        }, os.path.join(save_dir, save_ckpt_as_final))

    return {
        "model": model,
        "forward_process": forward_process,
        "block_size": block_size,
        "seed": seed,
        "T": T,
        "best_score": best_score,
        "best_loss": best_score,  # back-compat alias
        "best_epoch": best_epoch,
        "final_loss": metrics["loss"],
        "final_recon": metrics["recon"],
        "final_alphas": forward_process.get_alphas().detach().cpu().tolist(),
        "val_history": val_history,
        "select_by": select_by,
        "sigmoid_offset": sigmoid_offset,
        "loss_form": loss_form,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--block_size", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--sample_every", type=int, default=10)
    # new
    parser.add_argument("--sigmoid_offset", type=float,
                        default=LearnedForwardProcess.HISTORICAL_OFFSET,
                        help="Forward-process parameterization offset (B1). "
                             "Default -2 (historical, alpha floor ~0.0596). "
                             "Try -6 for floor ~0.00124.")
    parser.add_argument("--fixed_alphas", type=float, nargs="+", default=None,
                        help="If set, freezes the schedule to these alphas. "
                             "Length must equal T.")
    parser.add_argument("--loss_form", type=str, default="ce",
                        choices=["ce", "true_elbo"],
                        help="'ce' (historical) or 'true_elbo' (strict KL, B5).")
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Held-out validation fraction (B6).")
    parser.add_argument("--select_by", type=str, default="train_loss",
                        choices=["train_loss", "val_loss"],
                        help="Best-ckpt selection criterion (B6).")
    parser.add_argument("--val_samples_per_t", type=int, default=4,
                        help="z_t samples per t for val ELBO estimate.")
    parser.add_argument("--orientation", type=str, default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="For block_size=2: horizontal 1x2 (default) "
                             "or vertical 2x1 (B11 control).")
    args = parser.parse_args()

    result = run_mnist(
        block_size=args.block_size, seed=args.seed,
        T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, save_dir=args.save_dir,
        sample_every=args.sample_every, samples_dir="samples",
        verbose=True,
        sigmoid_offset=args.sigmoid_offset,
        fixed_alphas=args.fixed_alphas,
        loss_form=args.loss_form,
        val_fraction=args.val_fraction,
        select_by=args.select_by,
        val_samples_per_t=args.val_samples_per_t,
        orientation=args.orientation,
    )

    samples = sample(result["model"], result["forward_process"], args.T,
                     n_samples=64, device=args.device,
                     block_size=args.block_size, orientation=args.orientation)
    save_samples(samples, "samples/final.png")
    print("done. final samples saved to samples/final.png")


if __name__ == "__main__":
    main()
