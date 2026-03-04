"""Reusable plotting utilities for training and evaluation artifacts."""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PLOT_DIR, RESULT_DIR


# =======================
# Training Curves
# =======================

def save_training_plots(history: dict[str, Any]) -> None:
    """
    Save standard training plots from history dictionary.

    Parameters
    ----------
    history : dict[str, Any]
        History dictionary returned by trainer.train().
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    learning_rates = history["lr"]

    _plot_line(
        x=epochs,
        ys=[train_loss, val_loss],
        labels=["Train Loss", "Val Loss"],
        title="Training vs Validation Loss",
        y_label="Loss",
        output_path=os.path.join(PLOT_DIR, "loss_curve.png"),
    )

    _plot_line(
        x=epochs,
        ys=[train_acc, val_acc],
        labels=["Train Accuracy", "Val Accuracy"],
        title="Training vs Validation Accuracy",
        y_label="Accuracy (%)",
        output_path=os.path.join(PLOT_DIR, "accuracy_curve.png"),
    )

    _plot_line(
        x=epochs,
        ys=[learning_rates],
        labels=["Learning Rate"],
        title="Learning Rate Schedule",
        y_label="Learning Rate",
        output_path=os.path.join(PLOT_DIR, "learning_rate_curve.png"),
    )


def _plot_line(
    x: list[int],
    ys: list[list[float]],
    labels: list[str],
    title: str,
    y_label: str,
    output_path: str,
) -> None:
    """Create and save a line plot."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for values, label in zip(ys, labels):
        axis.plot(x, values, label=label)

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# =======================
# Confusion Matrix
# =======================

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[list[str]] = None,
    image_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix and save both image and CSV matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix of shape (N, N).
    class_names : list[str] | None
        Optional class labels.
    image_path : str | None
        Optional output path for PNG image.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    if image_path is None:
        image_path = os.path.join(RESULT_DIR, "confusion_matrix.png")

    classes = confusion_matrix.shape[0]
    figure, axis = plt.subplots(
        figsize=(max(8, classes * 0.18), max(8, classes * 0.18))
    )
    image = axis.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.set_title("Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    if class_names is not None and classes <= 20:
        ticks = np.arange(classes)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        axis.set_yticklabels(class_names, fontsize=7)

    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    figure.tight_layout()
    figure.savefig(image_path, dpi=150)
    plt.close(figure)

    np.savetxt(
        os.path.join(RESULT_DIR, "confusion_matrix.csv"),
        confusion_matrix,
        delimiter=",",
        fmt="%d",
    )
