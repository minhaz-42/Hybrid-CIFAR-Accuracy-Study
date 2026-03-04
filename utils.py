"""Shared utility helpers used across modules."""

from __future__ import annotations

import csv
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR


# =======================
# Environment Utilities
# =======================

def set_seed(seed: int) -> None:
    """
    Set random seed for deterministic behavior where possible.

    Parameters
    ----------
    seed : int
        Seed value used by Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device() -> str:
    """
    Detect available compute device.

    Returns
    -------
    str
        One of "mps", "cuda", or "cpu".
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_required_directories() -> None:
    """Create all required project output directories."""
    for path in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
        os.makedirs(path, exist_ok=True)


# =======================
# Exponential Moving Average
# =======================

class EMA:
    """
    Exponential moving average of trainable model parameters.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters are tracked.
    decay : float
        EMA decay coefficient.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.data.clone()

    def update(self, model: nn.Module) -> None:
        """
        Update EMA shadow parameters from current model parameters.

        Parameters
        ----------
        model : nn.Module
            Model containing latest parameters.
        """
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.data,
                    alpha=1.0 - self.decay,
                )

    def apply(self, model: nn.Module) -> None:
        """
        Apply EMA shadow parameters to model.

        Parameters
        ----------
        model : nn.Module
            Model to overwrite with EMA weights.
        """
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.backup[name] = parameter.data.clone()
                parameter.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """
        Restore model parameters previously backed up by apply().

        Parameters
        ----------
        model : nn.Module
            Model to restore.
        """
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


# =======================
# Checkpointing
# =======================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    filename: str,
) -> str:
    """
    Save checkpoint file to checkpoints directory.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer state.
    epoch : int
        Epoch number stored in checkpoint.
    best_acc : float
        Best validation accuracy tracked so far.
    filename : str
        File name for checkpoint.

    Returns
    -------
    str
        Path to saved checkpoint.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        checkpoint_path,
    )
    return checkpoint_path


# =======================
# Logging
# =======================

class CSVLogger:
    """
    Epoch-level CSV logger for training metrics.

    Parameters
    ----------
    filepath : str | None
        Optional custom path to CSV file.
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
        with open(self.filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: dict) -> None:
        """
        Write one row to CSV metrics log.

        Parameters
        ----------
        row : dict
            Row dictionary matching logger field names.
        """
        with open(self.filepath, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)
