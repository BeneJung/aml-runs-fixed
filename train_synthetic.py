"""E1: train FLDD on the synthetic 2x2-block dataset.

Logs final reconstruction loss and block-state TV distance between
generated samples and ground truth. Single-run CLI; see run_e1.py for the
sweep over block sizes and seeds.
"""

import argparse
import os
import torch
from tqdm import tqdm

from fldd.synthetic import (
    get_synthetic_dataset,
    empirical_block_dist,
    tv_distance,
)
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.train import train_epoch
from fldd.sample import sample, save_samples


def run_synthetic(block_size, seed, T=4, epochs=50, batch_size=128, lr=3e-4,
                  n_train=20000, n_test=5000, n_eval=5000, epsilon=0.04,
                  data_seed=0, device="cpu", save_dir="checkpoints_synth",
                  save_samples_path=None, verbose=True):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    train_loader, _, true_dist = get_synthetic_dataset(
        n_train=n_train, n_test=n_test, batch_size=batch_size,
        seed=data_seed, epsilon=epsilon,
    )

    forward_process = LearnedForwardProcess(T=T).to(device)
    model = UNet(channels=(32, 64, 128), block_size=block_size).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forward_process.parameters()),
        lr=lr,
    )

    last_metrics = None
    pbar = tqdm(range(1, epochs + 1), desc=f"|G|={block_size} seed={seed}",
                disable=not verbose)
    for _ in pbar:
        last_metrics = train_epoch(
            model, forward_process, train_loader, optimizer, T, device, block_size
        )
        if verbose:
            pbar.set_postfix(recon=f"{last_metrics['recon']:.3f}",
                             loss=f"{last_metrics['loss']:.3f}")

    samples = sample(model, forward_process, T, n_samples=n_eval,
                     device=device, block_size=block_size)
    emp = empirical_block_dist(samples.detach().cpu(), n_states=16)
    tv = tv_distance(emp, true_dist)

    if save_samples_path is not None:
        save_samples(samples[:64].cpu(), save_samples_path)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"bs{block_size}_s{seed}.pt")
    torch.save({
        "model": model.state_dict(),
        "forward": forward_process.state_dict(),
        "block_size": block_size,
        "T": T,
        "final_recon": last_metrics["recon"],
        "final_loss": last_metrics["loss"],
        "block_tv": tv,
        "true_dist": true_dist,
        "empirical_dist": emp,
        "seed": seed,
        "data_seed": data_seed,
        "epsilon": epsilon,
    }, ckpt_path)

    return {
        "block_size": block_size,
        "seed": seed,
        "final_recon": last_metrics["recon"],
        "final_loss": last_metrics["loss"],
        "block_tv": tv,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=5000)
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_synth")
    parser.add_argument("--save_samples", type=str, default=None,
                        help="optional path to save 64-sample grid")
    args = parser.parse_args()

    print(f"E1 synthetic | T={args.T} block_size={args.block_size} seed={args.seed}")
    result = run_synthetic(
        block_size=args.block_size, seed=args.seed,
        T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_train=args.n_train, n_test=args.n_test, n_eval=args.n_eval,
        epsilon=args.epsilon, data_seed=args.data_seed,
        device=args.device, save_dir=args.save_dir,
        save_samples_path=args.save_samples,
    )
    print(f"\nfinal | recon={result['final_recon']:.4f} | "
          f"loss={result['final_loss']:.4f} | block-TV={result['block_tv']:.4f}")


if __name__ == "__main__":
    main()
