"""Training entry point for binarized MNIST (v2-audit-revised).

Changes vs the pre-audit version:

* `loss_form` default is now **`"true_elbo"`** (B5 fix). Pass
  `--loss_form ce` to reproduce the historical loss values exactly.
* `--ema_decay` flag adds an EMA copy of the U-Net weights; EMA shadow is
  saved alongside live weights in every checkpoint. EMA-weighted forward
  computation is used for validation, sampling, and FID scoring.
* Three checkpoint kinds are written every selection event:
    - `<prefix>_best.pt`    — lowest training loss (legacy criterion)
    - `<prefix>_valbest.pt` — lowest val ELBO (B6; requires val_fraction>0)
    - `<prefix>_final.pt`   — last epoch
  Selecting any one for FID is then a post-hoc choice; no need to retrain
  to swap criteria. The `restore_into_model` argument controls which
  checkpoint the *returned* model is rolled back to (default: valbest if
  val is on, else best). EMA weights are stored in the same checkpoint
  file under the `"ema"` key.
* Validation ELBO uses a seeded `torch.Generator` (deterministic across
  epochs and runs).
"""

import argparse
import os

import torch
from tqdm import tqdm

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.train import (
    EMA,
    compute_validation_elbo,
    train_epoch,
    use_ema,
)
from fldd.sample import sample, save_samples


def _save_ckpt(path, *, epoch, model, forward_process, optimizer, ema,
               block_size, T, seed, sigmoid_offset, fixed_alphas, loss_form,
               select_by, train_loss, val_loss):
    """Write a checkpoint with model + forward + (optional) EMA + metadata."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "forward": forward_process.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": train_loss,
        "val_loss": val_loss,
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
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def run_mnist(
    block_size=1,
    seed=42,
    T=4,
    epochs=100,
    batch_size=128,
    lr=3e-4,
    device="cuda",
    save_dir="checkpoints",
    save_prefix="",   # if non-empty, ckpts named "<prefix>_{best,valbest,final}.pt"
    sample_every=10,
    samples_dir="samples",
    verbose=True,
    # ----- v2 audit options -----
    sigmoid_offset=LearnedForwardProcess.HISTORICAL_OFFSET,  # B1
    fixed_alphas=None,                                       # B4
    loss_form="true_elbo",                                   # B5 (default flipped)
    val_fraction=0.0,                                        # B6
    val_samples_per_t=1,                                     # B6
    val_seed=0,                                              # B6 (reproducibility)
    split_seed=0,                                            # B6
    orientation="horizontal",                                # B11
    ema_decay=None,                                          # NEW: float in (0, 1) or None
    restore_into_model="auto",
    # ^ "auto" -> valbest if val on else best; "valbest" | "best" | "final" | None
):
    """Train FLDD on binarized MNIST.

    Returns a dict with model, forward process, EMA (or None), per-criterion
    best epochs/losses, val history, and the alphas.

    The training pipeline writes up to three checkpoints per run:
        <save_dir>/<save_prefix>_best.pt     (lowest train loss)
        <save_dir>/<save_prefix>_valbest.pt  (lowest val ELBO)   [if val on]
        <save_dir>/<save_prefix>_final.pt    (last epoch)
    Each contains live U-Net + forward weights and, if `ema_decay` is set,
    the EMA shadow under key `"ema"`.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    use_val = val_fraction > 0.0
    use_ema_flag = ema_decay is not None and ema_decay > 0.0

    if verbose:
        print(
            f"Training FLDD | T={T} block_size={block_size} seed={seed} "
            f"device={device}"
        )
        print(
            f"  sigmoid_offset={sigmoid_offset}  "
            f"fixed_alphas={'set' if fixed_alphas is not None else 'no'}  "
            f"loss_form={loss_form}  val_fraction={val_fraction}  "
            f"ema_decay={ema_decay}"
        )

    if use_val:
        train_loader, val_loader, _ = get_binarized_mnist(
            batch_size=batch_size, val_fraction=val_fraction, split_seed=split_seed,
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
        channels=(32, 64, 128), block_size=block_size, orientation=orientation,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  U-Net parameters: {n_params:,}")
        print(f"  alpha floor (param.): {forward_process.alpha_floor:.6f}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forward_process.parameters()), lr=lr,
    )
    ema = EMA(model, decay=ema_decay) if use_ema_flag else None

    os.makedirs(save_dir, exist_ok=True)
    if sample_every and samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    prefix = save_prefix if save_prefix else "ckpt"
    path_best    = os.path.join(save_dir, f"{prefix}_best.pt")
    path_valbest = os.path.join(save_dir, f"{prefix}_valbest.pt")
    path_final   = os.path.join(save_dir, f"{prefix}_final.pt")

    best_train_loss = float("inf"); best_train_epoch = None
    best_val_loss   = float("inf"); best_val_epoch   = None
    metrics = None
    val_metrics = None
    val_history = []

    save_kwargs_common = dict(
        block_size=block_size, T=T, seed=seed,
        sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas,
        loss_form=loss_form,
    )

    for epoch in tqdm(range(1, epochs + 1), desc="training", disable=not verbose):
        metrics = train_epoch(
            model, forward_process, train_loader, optimizer,
            T, device, block_size, loss_form=loss_form,
            orientation=orientation, ema=ema,
        )

        # Validation runs on EMA weights when EMA is on — that's what we'll
        # FID-score downstream, so val should reflect the same model.
        if use_val:
            with use_ema(model, ema):
                val_metrics = compute_validation_elbo(
                    model, forward_process, val_loader, T, device,
                    block_size=block_size, loss_form=loss_form,
                    samples_per_t=val_samples_per_t,
                    orientation=orientation, seed=val_seed,
                )
            val_history.append({"epoch": epoch, **val_metrics})

        alphas = forward_process.get_alphas().detach().cpu().tolist()
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
            line += f" | alphas [{', '.join(f'{a:.4f}' for a in alphas)}]"
            print(line)

        # --- save best (by train loss) ---
        if metrics["loss"] < best_train_loss:
            best_train_loss = metrics["loss"]
            best_train_epoch = epoch
            _save_ckpt(
                path_best, epoch=epoch, model=model,
                forward_process=forward_process, optimizer=optimizer, ema=ema,
                select_by="train_loss",
                train_loss=metrics["loss"],
                val_loss=val_metrics["val_loss"] if val_metrics else None,
                **save_kwargs_common,
            )

        # --- save valbest (by val ELBO) ---
        if val_metrics is not None and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_val_epoch = epoch
            _save_ckpt(
                path_valbest, epoch=epoch, model=model,
                forward_process=forward_process, optimizer=optimizer, ema=ema,
                select_by="val_loss",
                train_loss=metrics["loss"],
                val_loss=val_metrics["val_loss"],
                **save_kwargs_common,
            )

        if (sample_every and sample_every > 0 and samples_dir
                and epoch % sample_every == 0):
            with use_ema(model, ema):
                samples = sample(
                    model, forward_process, T, n_samples=64,
                    device=device, block_size=block_size, orientation=orientation,
                )
            save_samples(samples, os.path.join(samples_dir, f"epoch_{epoch:03d}.png"))

    # --- save final ---
    _save_ckpt(
        path_final, epoch=epochs, model=model,
        forward_process=forward_process, optimizer=optimizer, ema=ema,
        select_by="final",
        train_loss=metrics["loss"],
        val_loss=val_metrics["val_loss"] if val_metrics else None,
        **save_kwargs_common,
    )

    # Decide which ckpt to load into the returned model
    if restore_into_model == "auto":
        restore_into_model = "valbest" if use_val else "best"
    restore_path = {
        "valbest": path_valbest, "best": path_best, "final": path_final,
    }.get(restore_into_model)
    restored = None
    if restore_path is not None and os.path.exists(restore_path):
        ckpt = torch.load(restore_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        forward_process.load_state_dict(ckpt["forward"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], device=device)
        restored = {"which": restore_into_model, "epoch": ckpt["epoch"]}
        if verbose:
            print(f"  restored {restore_into_model} (epoch {ckpt['epoch']}) "
                  f"into returned model")

    return {
        "model": model,
        "forward_process": forward_process,
        "ema": ema,
        "block_size": block_size,
        "seed": seed,
        "T": T,
        "best_train_loss": best_train_loss,
        "best_train_epoch": best_train_epoch,
        "best_val_loss": best_val_loss if best_val_epoch is not None else None,
        "best_val_epoch": best_val_epoch,
        "final_loss": metrics["loss"],
        "final_recon": metrics["recon"],
        "final_alphas": forward_process.get_alphas().detach().cpu().tolist(),
        "val_history": val_history,
        "sigmoid_offset": sigmoid_offset,
        "loss_form": loss_form,
        "ema_decay": ema_decay,
        "restored": restored,
        "ckpt_paths": {"best": path_best, "valbest": path_valbest, "final": path_final},
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
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--block_size", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--sigmoid_offset", type=float,
                        default=LearnedForwardProcess.HISTORICAL_OFFSET)
    parser.add_argument("--fixed_alphas", type=float, nargs="+", default=None)
    parser.add_argument("--loss_form", type=str, default="true_elbo",
                        choices=["ce", "true_elbo"])
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--val_samples_per_t", type=int, default=1)
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--orientation", type=str, default="horizontal",
                        choices=["horizontal", "vertical"])
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay in (0, 1). Recommended 0.9999. "
                             "Default off, for backwards compat.")
    parser.add_argument("--restore", type=str, default="auto",
                        choices=["auto", "valbest", "best", "final", "none"])
    args = parser.parse_args()

    restore = None if args.restore == "none" else args.restore

    result = run_mnist(
        block_size=args.block_size, seed=args.seed,
        T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, save_dir=args.save_dir, save_prefix=args.save_prefix,
        sample_every=args.sample_every, samples_dir="samples",
        verbose=True,
        sigmoid_offset=args.sigmoid_offset,
        fixed_alphas=args.fixed_alphas,
        loss_form=args.loss_form,
        val_fraction=args.val_fraction,
        val_samples_per_t=args.val_samples_per_t,
        val_seed=args.val_seed,
        orientation=args.orientation,
        ema_decay=args.ema_decay,
        restore_into_model=restore,
    )

    samples = sample(result["model"], result["forward_process"], args.T,
                     n_samples=64, device=args.device,
                     block_size=args.block_size, orientation=args.orientation)
    save_samples(samples, "samples/final.png")
    print("done. final samples saved to samples/final.png")


if __name__ == "__main__":
    main()
