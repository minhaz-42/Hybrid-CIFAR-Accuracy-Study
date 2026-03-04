"""Evaluation logic for validation and test-time reporting."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULT_DIR, Config
from data import get_class_names
from visualization import plot_confusion_matrix


# =======================
# Validation
# =======================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Validate model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : str
        Compute device.

    Returns
    -------
    tuple[float, float]
        Average loss and top-1 accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


# =======================
# Evaluation Metrics
# =======================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Run complete evaluation with top-1 and top-5 metrics.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Evaluation data loader.
    device : str
        Compute device.

    Returns
    -------
    tuple[float, float, float, np.ndarray, np.ndarray]
        Loss, top-1 accuracy, top-5 accuracy, predictions, and targets.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

        top1 = outputs.argmax(dim=1)
        _, top5 = outputs.topk(5, dim=1, largest=True, sorted=True)

        total += targets.size(0)
        top1_correct += top1.eq(targets).sum().item()
        top5_correct += top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

        all_preds.extend(top1.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / total
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    return avg_loss, top1_acc, top5_acc, np.array(all_preds), np.array(all_targets)


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    preds : np.ndarray
        Predicted labels.
    targets : np.ndarray
        True labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(targets, preds):
        matrix[true_label, pred_label] += 1
    return matrix


def _save_metrics_file(
    top1: float,
    top5: float,
    loss: float,
    cfg: Config,
    best_val_acc: float,
) -> None:
    """Save final metrics summary to results directory."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "metrics.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"=== Evaluation Results — {cfg.dataset.upper()} ===\n\n")
        file.write(f"Top-1 Accuracy: {top1:.2f}%\n")
        file.write(f"Top-5 Accuracy: {top5:.2f}%\n")
        file.write(f"Validation Loss: {loss:.4f}\n")
        file.write(f"Best Validation Accuracy (training): {best_val_acc:.2f}%\n")
        file.write(f"Epochs: {cfg.epochs}\n")
        file.write(f"Batch Size: {cfg.batch_size}\n")
        file.write("Optimizer: AdamW\n")
        file.write("Scheduler: OneCycleLR\n")
        file.write(f"EMA Decay: {cfg.ema_decay}\n")
        file.write(f"AMP Enabled: {bool(cfg.use_amp and cfg.device == 'cuda')}\n")


# =======================
# Best Checkpoint Evaluation
# =======================

def evaluate_best_model(model: nn.Module, val_loader: DataLoader, cfg: Config) -> dict[str, Any]:
    """
    Evaluate the best checkpoint and save confusion-matrix artifacts.

    Parameters
    ----------
    model : nn.Module
        Model instance used for loading checkpoint and evaluation.
    val_loader : DataLoader
        Validation loader.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    dict[str, Any]
        Evaluation dictionary with loss/accuracy metrics.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)

    loss, top1, top5, preds, targets = evaluate(model, val_loader, cfg.device)
    confusion_matrix = compute_confusion_matrix(preds, targets, cfg.num_classes)

    class_names = get_class_names(cfg.dataset)
    plot_confusion_matrix(confusion_matrix, class_names=class_names)

    best_val_acc = float(checkpoint.get("best_acc", top1))
    _save_metrics_file(top1, top5, loss, cfg, best_val_acc)

    return {
        "val_loss": loss,
        "top1": top1,
        "top5": top5,
        "best_val_acc": best_val_acc,
    }
