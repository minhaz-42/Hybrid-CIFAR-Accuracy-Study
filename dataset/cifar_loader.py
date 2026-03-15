"""CIFAR-10 data loading, augmentation, and train/validation splitting."""

from __future__ import annotations

import os
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.0,
    use_randaugment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 training and test data loaders.

    Parameters
    ----------
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    val_split : float
        If > 0, split this fraction from the training set for validation.
        Otherwise the test set is used for validation.
    use_randaugment : bool
        Whether to include RandAugment in training transforms.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        train_transforms.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform,
    )

    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset_raw = random_split(train_dataset, [train_size, val_size])
        # Apply val transform to the validation subset
        val_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=True, download=False, transform=val_transform,
        )
        val_dataset_raw.dataset = val_dataset
        val_dataset = val_dataset_raw
    else:
        val_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=val_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
