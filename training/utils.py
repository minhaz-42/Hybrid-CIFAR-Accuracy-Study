"""Shared utility helpers: seed, device detection, EMA, checkpointing, CSV logging."""

from __future__ import annotations

import csv
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
RESULT_DIR = os.path.join(BASE_DIR, "results")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device() -> str:
    """Detect best available compute device: mps > cuda > cpu."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dirs() -> None:
    """Create all required output directories."""
    for path in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
        os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, best_acc, filename, experiment_name=""):
    """Save a training checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "experiment": experiment_name,
    }, path)
    return path


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class CSVLogger:
    """Epoch-level CSV logger for training metrics."""

    def __init__(self, filepath: str | None = None):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.filepath = filepath or os.path.join(LOG_DIR, "training_log.csv")
        self.fieldnames = [
            "experiment", "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "learning_rate",
        ]
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: dict) -> None:
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
