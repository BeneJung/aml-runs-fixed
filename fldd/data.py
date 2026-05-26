import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


def get_binarized_mnist(
    data_dir="./data",
    batch_size=128,
    num_workers=0,
    val_fraction=0.0,
    split_seed=0,
):
    """Load MNIST and binarize by thresholding at 0.5.

    Returns binary tensors (B, 1, 28, 28) in {0, 1}.

    Args:
        val_fraction: if > 0, hold out this fraction of the train set as
            a validation split. The returned tuple becomes
            (train_loader, val_loader, test_loader). When 0 (default,
            original behaviour), the tuple is (train_loader, test_loader)
            for backward compat with existing scripts.
        split_seed: seed for the train/val split permutation. Fixing this
            keeps the split identical across runs / block sizes / seeds.
    """
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # binarize: threshold at 0.5
    train_imgs = (train_set.data.float() / 255.0 > 0.5).float()
    test_imgs = (test_set.data.float() / 255.0 > 0.5).float()
    train_imgs = train_imgs.unsqueeze(1)
    test_imgs = test_imgs.unsqueeze(1)

    test_loader = DataLoader(
        TensorDataset(test_imgs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if val_fraction <= 0.0:
        train_loader = DataLoader(
            TensorDataset(train_imgs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        return train_loader, test_loader

    # held-out validation split (B6)
    n = train_imgs.shape[0]
    n_val = int(round(n * val_fraction))
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(train_imgs[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(train_imgs[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
