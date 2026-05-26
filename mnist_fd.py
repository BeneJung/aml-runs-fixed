"""Frechet distance for binarized MNIST using MNIST-classifier features.

Standard FID uses InceptionV3 trained on ImageNet, whose features are
poorly calibrated for single-channel, hard-thresholded MNIST. As a
companion metric we train a small MNIST classifier once, freeze it,
and report the Frechet distance between real and generated samples
in its penultimate-layer feature space.

The classifier achieves >99% test accuracy in a minute on CPU, and the
resulting metric is more sensitive to MNIST-specific failure modes than
ImageNet InceptionV3.

Usage
-----

Train the classifier once (cached to ~/.cache/fldd_mnist_clf.pt):

    python mnist_fd.py --train_classifier

Compute MNIST-FD between two image directories:

    python mnist_fd.py \\
        --real_dir fid_stats/real --gen_dir fid_stats_e2/bs4_s42

Or score a set of E2 checkpoints in one pass:

    python mnist_fd.py --score_ckpts checkpoints_e2 --device cuda \\
        --results_json results/mnist_fd_e2.json

Outputs a scalar per (real, gen) pair plus an optional JSON.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CACHE_PATH = os.path.expanduser("~/.cache/fldd_mnist_clf.pt")
FEATURE_DIM = 128


class MnistClassifier(nn.Module):
    """Small CNN classifier on binarized MNIST.

    Penultimate-layer feature dim = 128 (the input to the final FC).
    """

    def __init__(self, n_classes=10, feature_dim=FEATURE_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, feature_dim)
        self.fc2 = nn.Linear(feature_dim, n_classes)

    def features(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2)            # 14x14
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)            # 7x7
        h = F.relu(self.conv3(h))
        h = F.max_pool2d(h, 2)            # 3x3
        h = h.flatten(1)
        feat = F.relu(self.fc1(h))
        return feat

    def forward(self, x):
        return self.fc2(self.features(x))


def train_classifier(device="cpu", epochs=3, batch_size=128, lr=1e-3,
                     cache_path=CACHE_PATH, data_dir="./data"):
    """Train and cache the classifier. ~1 min on CPU for 3 epochs.

    Trained on hard-thresholded binarized MNIST so the features see the
    same distribution as the generated samples we'll score against.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0.5).float()),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True,
                              transform=transform)
    tl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    te = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    clf = MnistClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        clf.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(clf(x), y)
            loss.backward()
            opt.step()
        # eval
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(device), y.to(device)
                correct += (clf(x).argmax(-1) == y).sum().item()
                total += y.numel()
        print(f"  epoch {ep}: test acc = {correct/total*100:.2f}%")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(clf.state_dict(), cache_path)
    print(f"cached classifier weights -> {cache_path}")
    return clf


def load_or_train_classifier(device="cpu", cache_path=CACHE_PATH):
    clf = MnistClassifier().to(device)
    if os.path.exists(cache_path):
        clf.load_state_dict(torch.load(cache_path, map_location=device,
                                       weights_only=True))
        clf.eval()
        return clf
    print(f"no cached classifier at {cache_path}; training...")
    return train_classifier(device=device, cache_path=cache_path)


@torch.no_grad()
def extract_features(clf, image_paths, device, batch_size=128):
    """Read PNGs from `image_paths` and return (N, FEATURE_DIM) tensor.

    Images are expected to be single-channel hard-thresholded MNIST-shaped
    PNGs (28x28). pytorch_fid's generate-then-save pipeline produces
    3-channel replications; we collapse those back to 1 channel here.
    """
    from PIL import Image
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            arr = (arr > 0.5).astype(np.float32)
            imgs.append(torch.from_numpy(arr))
        x = torch.stack(imgs).unsqueeze(1).to(device)  # (B, 1, 28, 28)
        feats.append(clf.features(x).cpu())
    return torch.cat(feats, dim=0)


def frechet_distance(features_a, features_b):
    """Frechet distance between two Gaussian-fit feature distributions.

    Numerically stable matrix sqrt via scipy.linalg.sqrtm.
    """
    from scipy.linalg import sqrtm

    mu_a = features_a.mean(dim=0).numpy()
    mu_b = features_b.mean(dim=0).numpy()
    sigma_a = np.cov(features_a.numpy(), rowvar=False)
    sigma_b = np.cov(features_b.numpy(), rowvar=False)

    diff = mu_a - mu_b
    # matrix product, then square root
    covmean = sqrtm(sigma_a.dot(sigma_b))
    if np.iscomplexobj(covmean):
        # tiny imaginary parts from numerical noise
        covmean = covmean.real
    fd = diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean)
    return float(fd)


def list_pngs(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_classifier", action="store_true",
                        help="(re)train + cache the MNIST classifier and exit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--score_ckpts", type=str, default=None,
                        help="if set, scores every <ckpt_dir>/bs*_s*_best.pt "
                             "by sampling 10k images, computing MNIST-FD vs "
                             "the MNIST test set, and dumps a JSON.")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--results_json", type=str, default=None)
    args = parser.parse_args()

    if args.train_classifier:
        train_classifier(device=args.device, epochs=args.epochs)
        return

    clf = load_or_train_classifier(device=args.device)

    if args.real_dir and args.gen_dir:
        real_paths = list_pngs(args.real_dir)
        gen_paths = list_pngs(args.gen_dir)
        if not real_paths or not gen_paths:
            print(f"missing pngs (real={len(real_paths)} gen={len(gen_paths)})",
                  file=sys.stderr)
            sys.exit(2)
        f_real = extract_features(clf, real_paths, args.device)
        f_gen = extract_features(clf, gen_paths, args.device)
        fd = frechet_distance(f_real, f_gen)
        print(f"MNIST-FD ({len(real_paths)} real, {len(gen_paths)} gen): "
              f"{fd:.4f}")
        return

    if args.score_ckpts:
        # batch-score E2-style checkpoints
        from fldd.forward import LearnedForwardProcess
        from fldd.unet import UNet
        from run_e2 import (
            ensure_real_fid_images, generate_samples_to_dir,
        )
        import shutil

        real_dir = "fid_stats/real"
        ensure_real_fid_images(real_dir)
        real_paths = list_pngs(real_dir)
        f_real = extract_features(clf, real_paths, args.device)

        CKPT_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")
        out = []
        for path in sorted(glob.glob(os.path.join(args.score_ckpts,
                                                  "*_best.pt"))):
            m = CKPT_RE.search(os.path.basename(path))
            if not m:
                continue
            bs, seed = int(m.group(1)), int(m.group(2))
            ckpt = torch.load(path, map_location=args.device, weights_only=False)
            T = ckpt["T"]
            sigmoid_offset = ckpt.get(
                "sigmoid_offset", LearnedForwardProcess.HISTORICAL_OFFSET,
            )
            model = UNet(channels=(32, 64, 128), block_size=bs).to(args.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            fp = LearnedForwardProcess(T=T, sigmoid_offset=sigmoid_offset).to(args.device)
            fp.load_state_dict(ckpt["forward"])
            fp.eval()

            gen_dir = f"/tmp/mnist_fd_{bs}_{seed}"
            generate_samples_to_dir(model, fp, T, bs, args.n_samples,
                                    gen_dir, args.device)
            gen_paths = list_pngs(gen_dir)
            f_gen = extract_features(clf, gen_paths, args.device)
            fd = frechet_distance(f_real, f_gen)
            print(f"|G|={bs} seed={seed}: MNIST-FD={fd:.4f}")
            out.append({
                "block_size": bs, "seed": seed,
                "ckpt": path, "mnist_fd": fd, "n_samples": args.n_samples,
            })
            shutil.rmtree(gen_dir, ignore_errors=True)

        if args.results_json:
            os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump({"per_run": out}, f, indent=2)
            print(f"wrote {args.results_json}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
