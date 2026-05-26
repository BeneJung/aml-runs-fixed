"""Compute FID between generated samples and real binarized MNIST."""

import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.sample import sample


def save_images_for_fid(images, out_dir):
    """Save individual images as PNGs for pytorch-fid."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, os.path.join(out_dir, f"{i:05d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--real_dir", type=str, default="fid_stats/real")
    parser.add_argument("--gen_dir", type=str, default="fid_stats/generated")
    args = parser.parse_args()

    # load model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    block_size = ckpt.get("block_size", 1)
    T = ckpt.get("T", args.T)
    model = UNet(channels=(32, 64, 128), block_size=block_size).to(args.device)
    model.load_state_dict(ckpt["model"])
    forward_process = LearnedForwardProcess(T=T).to(args.device)
    forward_process.load_state_dict(ckpt["forward"])

    print(f"loaded checkpoint from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}, block_size={block_size}")
    print(f"alphas: {forward_process.get_alphas().detach().cpu().tolist()}")

    # generate samples
    print(f"generating {args.n_samples} samples...")
    all_samples = []
    remaining = args.n_samples
    while remaining > 0:
        n = min(args.batch_size, remaining)
        s = sample(model, forward_process, T, n_samples=n, device=args.device, block_size=block_size)
        # convert to 3-channel for FID (InceptionV3 expects RGB)
        s_rgb = s.repeat(1, 3, 1, 1)
        all_samples.append(s_rgb.cpu())
        remaining -= n

    all_samples = torch.cat(all_samples, dim=0)
    print(f"saving generated images to {args.gen_dir}")
    save_images_for_fid(all_samples, args.gen_dir)

    # save real images if they don't exist yet
    if not os.path.exists(args.real_dir) or len(os.listdir(args.real_dir)) == 0:
        print(f"saving real images to {args.real_dir}")
        _, test_loader = get_binarized_mnist(batch_size=256)
        real_imgs = []
        for (x,) in test_loader:
            real_imgs.append(x.repeat(1, 3, 1, 1))
        real_imgs = torch.cat(real_imgs, dim=0)
        save_images_for_fid(real_imgs, args.real_dir)

    print("\nto compute FID, run:")
    print(f"  python -m pytorch_fid {args.real_dir} {args.gen_dir}")


if __name__ == "__main__":
    main()
