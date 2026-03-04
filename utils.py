"""
utils.py — Utility functions for data loading, EMA, logging, evaluation,
and visualisation.

This module houses every helper that is *not* the model or the training
loop, keeping the project cleanly factored.
"""

from __future__ import annotations

import copy
import csv
import os
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOT_DIR,
    RESULT_DIR,
    Config,
)


# ======================================================================== #
#  Reproducibility                                                          #
# ======================================================================== #

def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure
    reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================== #
#  Data Loading                                                             #
# ======================================================================== #

# CIFAR normalisation statistics (channel-wise mean and std)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation ``DataLoader`` instances for the
    chosen CIFAR variant.

    Training augmentation
    ~~~~~~~~~~~~~~~~~~~~~
    * ``RandomCrop(32, padding=4)``
    * ``RandomHorizontalFlip``
    * ``RandAugment(num_ops=2, magnitude=9)``
    * Channel-wise normalisation

    Validation applies only normalisation.

    Parameters
    ----------
    cfg : Config
        Project configuration.

    Returns
    -------
    train_loader, val_loader : tuple[DataLoader, DataLoader]
    """
    if cfg.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        dataset_cls = datasets.CIFAR10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        dataset_cls = datasets.CIFAR100

    # --- augmentation pipeline ---
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = dataset_cls(
        root=DATA_DIR, train=True, download=True, transform=train_transform,
    )
    val_dataset = dataset_cls(
        root=DATA_DIR, train=False, download=True, transform=val_transform,
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


# ======================================================================== #
#  Exponential Moving Average (EMA)                                         #
# ======================================================================== #

class EMA:
    """
    Maintain an exponential moving average of model parameters.

    After each optimiser step, call ``update()`` to track the running
    average.  For evaluation, call ``apply()`` to copy EMA weights into
    the model and ``restore()`` to revert.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are tracked.
    decay : float
        EMA decay factor (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with the latest model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        """Replace model weights with EMA shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ======================================================================== #
#  Checkpointing                                                            #
# ======================================================================== #

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    filename: str = "best_model.pth",
) -> str:
    """
    Persist a training checkpoint to disk.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    epoch : int
    best_acc : float
    filename : str

    Returns
    -------
    str
        Path to the saved checkpoint file.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """
    Load a checkpoint and return the stored epoch and best accuracy.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["best_acc"]


# ======================================================================== #
#  CSV Logger                                                               #
# ======================================================================== #

class CSVLogger:
    """
    Append-mode CSV logger that writes one row per epoch.

        Columns:
            ``epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate``
    """

    def __init__(self, filepath: str | None = None):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.filepath = filepath or os.path.join(LOG_DIR, "training_log.csv")
        self.fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
        ]
        # Write header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: dict) -> None:
        """Append a single row (dict) to the CSV file."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ======================================================================== #
#  Evaluation helpers                                                       #
# ======================================================================== #

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Run a full evaluation pass and return metrics.

    Returns
    -------
    loss : float
    top1_acc : float
    top5_acc : float
    all_preds : np.ndarray
    all_targets : np.ndarray
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

        # Top-1
        _, pred = outputs.max(1)
        top1_correct += pred.eq(targets).sum().item()

        # Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        top5_correct += top5_pred.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / total
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return avg_loss, top1_acc, top5_acc, np.array(all_preds), np.array(all_targets)


# ======================================================================== #
#  Confusion Matrix                                                         #
# ======================================================================== #

def compute_confusion_matrix(
    preds: np.ndarray, targets: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Build a (num_classes × num_classes) confusion matrix.

    ``cm[i][j]`` = number of samples with true label *i* predicted as *j*.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


# ======================================================================== #
#  Visualisation / Plotting                                                 #
# ======================================================================== #

def plot_curves(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    lrs: List[float],
) -> None:
    """
    Generate and save three plots:
      1. Training vs Validation Loss
      2. Training vs Validation Accuracy
      3. Learning Rate Curve

    All plots are saved as PNG files in the ``plots/`` directory.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- Loss curve ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # ---- Accuracy curve ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, label="Train Acc")
    ax.plot(epochs, val_accs, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "accuracy_curve.png"), dpi=150)
    plt.close(fig)

    # ---- Learning rate curve ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lrs, label="Learning Rate", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "learning_rate_curve.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Plots saved to {PLOT_DIR}/")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] | None = None,
    image_path: str | None = None,
) -> None:
    """
    Render and save a confusion-matrix heat-map.

    For CIFAR-100, labels are omitted from axes to keep the figure
    legible; the raw matrix is still saved alongside the image.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    if image_path is None:
        image_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    num_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.18), max(8, num_classes * 0.18)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names is not None and num_classes <= 20:
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=7)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    # Also save the raw matrix as a text file for reference
    np.savetxt(os.path.join(RESULT_DIR, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    print(f"[INFO] Confusion matrix saved to {image_path}")


def save_metrics(
    top1: float,
    top5: float,
    dataset_name: str,
    extra: str = "",
) -> None:
    """
    Write final evaluation metrics to ``results/metrics.txt``.

    Parameters
    ----------
    top1 : float
        Top-1 accuracy (%).
    top5 : float
        Top-5 accuracy (%).
    dataset_name : str
        Name of the dataset (for the header line).
    extra : str
        Any additional text to append.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "metrics.txt")
    with open(path, "w") as f:
        f.write(f"=== Evaluation Results — {dataset_name.upper()} ===\n\n")
        f.write(f"Top-1 Accuracy: {top1:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5:.2f}%\n")
        if extra:
            f.write(f"\n{extra}\n")
    print(f"[INFO] Metrics saved to {path}")
