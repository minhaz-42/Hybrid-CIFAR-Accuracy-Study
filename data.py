"""Data pipeline utilities for CIFAR experiments."""

from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, Config


# =======================
# Dataset Statistics
# =======================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# =======================
# Data Loading
# =======================

def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation data loaders for CIFAR datasets.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Train and validation loaders.
    """
    if cfg.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        dataset_cls = datasets.CIFAR10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        dataset_cls = datasets.CIFAR100

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = dataset_cls(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = dataset_cls(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_class_names(dataset: str) -> Optional[list[str]]:
    """
    Return class names for axis labels when available.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    list[str] | None
        CIFAR-10 class names, or None for CIFAR-100.
    """
    if dataset == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    return None
