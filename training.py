"""
training.py — Main training & evaluation script for the Hybrid CNN-ViT.

Usage
-----
    # Train on CIFAR-10 (default)
    python training.py

    # Train on CIFAR-100
    python training.py --dataset cifar100

The script will:
    1. Prepare data loaders with strong augmentation.
    2. Build the Hybrid CNN-ViT model.
    3. Train for ``cfg.epochs`` epochs using AdamW + OneCycleLR + AMP + EMA.
    4. Log per-epoch metrics to ``logs/training_log.csv``.
    5. Save the best checkpoint to ``checkpoints/best_model.pth``.
    6. After training, evaluate with the EMA weights and produce:
        - Accuracy / loss / LR plots  →  ``plots/``
        - Top-1, Top-5 accuracy + confusion matrix  →  ``results/``
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR, Config
from model import build_model
from utils import (
    EMA,
    compute_confusion_matrix,
    evaluate,
    get_dataloaders,
    plot_confusion_matrix,
    save_checkpoint,
    save_metrics,
    set_seed,
)


# Enable fallback for unsupported MPS operations as requested.
if hasattr(torch.backends, "mps"):
    torch.backends.mps.enable_fallback = True


def detect_device() -> str:
    """
    Detect the best available device with priority: MPS -> CUDA -> CPU.

    Returns
    -------
    str
        One of ``"mps"``, ``"cuda"``, or ``"cpu"``.
    """
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def ensure_required_directories() -> None:
    """Create all required project directories before training starts."""
    for path in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
        os.makedirs(path, exist_ok=True)


def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    lrs: List[float],
) -> None:
    """Generate and save training curves with clean styling."""
    os.makedirs("plots", exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/accuracy_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.plot(epochs, lrs, label="Learning Rate")
    plt.title("Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/learning_rate_curve.png", dpi=300)
    plt.close()


# ======================================================================== #
#  Train one epoch                                                          #
# ======================================================================== #

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    """
    Execute a single training epoch.

    Parameters
    ----------
    model : nn.Module
        The model being trained.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function (CrossEntropyLoss).
    optimizer : Optimizer
        AdamW optimiser.
    scheduler : _LRScheduler
        OneCycleLR scheduler (stepped per batch).
    scaler : GradScaler
        AMP gradient scaler.
    ema : EMA
        Exponential moving average tracker.
    device : str
        ``'cuda'`` or ``'cpu'``.
    use_amp : bool
        Enable CUDA mixed precision for training.
    grad_clip_norm : float
        Maximum norm used for gradient clipping.

    Returns
    -------
    avg_loss : float
    accuracy : float (percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---- Mixed precision forward pass ----
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # ---- Backward pass with gradient scaling ----
        scaler.scale(loss).backward()

        # ---- Gradient clipping ----
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # ---- EMA update ----
        ema.update(model)

        # ---- Step the OneCycleLR scheduler (per batch) ----
        scheduler.step()

        # ---- Bookkeeping ----
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ======================================================================== #
#  Validation pass                                                          #
# ======================================================================== #

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Run one validation epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float (percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ======================================================================== #
#  Main training loop                                                       #
# ======================================================================== #

def main(cfg: Config) -> None:
    """
    Full training pipeline:
      seed → data → model → train → evaluate → save results.
    """
    # ---- Ensure required folders exist ----
    ensure_required_directories()
    os.makedirs("logs", exist_ok=True)
    full_output_log_path = "logs/full_training_output.txt"
    final_summary_path = os.path.join(RESULT_DIR, "final_summary.txt")
    training_csv_path = "logs/training_log.csv"

    # Reset epoch-by-epoch plain text training output for this run.
    open(full_output_log_path, "w").close()

    if not os.path.exists(training_csv_path):
        with open(training_csv_path, "w") as f:
            f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate\n")

    # ---- Reproducibility ----
    set_seed(cfg.seed)
    cfg.device = detect_device()
    use_amp = cfg.device == "cuda"

    print(f"\n{'='*60}")
    print(f"  Hybrid CNN-ViT  |  {cfg.dataset.upper()}  |  device: {cfg.device}")
    print(f"{'='*60}\n")

    # ---- Data ----
    train_loader, val_loader = get_dataloaders(cfg)
    total_epochs = cfg.epochs
    steps_per_epoch = len(train_loader)
    print(f"[DATA] Train batches: {steps_per_epoch}  |  Val batches: {len(val_loader)}")

    # ---- Model ----
    model = build_model(cfg).to(cfg.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {num_params:,}")

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss()

    # ---- Optimiser: AdamW ----
    adamw_lr = 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adamw_lr,
        weight_decay=0.05,
    )

    # ---- Scheduler: OneCycleLR ----
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
    )

    # ---- AMP scaler ----
    scaler = GradScaler(enabled=use_amp)

    # ---- EMA ----
    ema = EMA(model, decay=cfg.ema_decay)

    # ---- Tracking lists for plots ----
    epoch_list: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    lr_list: List[float] = []
    history: List[Dict[str, float]] = []

    best_val_acc = 0.0

    # ================================================================== #
    #  Training loop                                                      #
    # ================================================================== #
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, cfg.device, use_amp, cfg.grad_clip_norm,
        )

        # ---- Validate with EMA weights ----
        ema.apply(model)
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.device)
        ema.restore(model)

        # ---- Current learning rate ----
        current_lr = scheduler.get_last_lr()[0]

        # ---- Book-keeping ----
        epoch_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lr_list.append(current_lr)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }
        )

        with open("logs/training_log.csv", "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},"
                f"{val_acc:.2f},{current_lr:.6f}\n"
            )

        # ---- Checkpoint best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema.apply(model)
            save_checkpoint(model, optimizer, epoch, best_val_acc, "best_model.pth")
            ema.restore(model)

        formatted_epoch_output = (
            f"Epoch [{epoch}/{total_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        print(formatted_epoch_output, flush=True)
        with open(full_output_log_path, "a") as f:
            f.write(formatted_epoch_output + "\n")

    # ---- Save last model checkpoint ----
    save_checkpoint(model, optimizer, total_epochs, best_val_acc, "last_model.pth")

    # ---- Save EMA model separately ----
    torch.save(
        {
            "epoch": total_epochs,
            "ema_state_dict": ema.shadow,
            "best_val_acc": best_val_acc,
        },
        os.path.join(CHECKPOINT_DIR, "ema_model.pth"),
    )

    # ================================================================== #
    #  Post-training evaluation                                           #
    # ================================================================== #
    print(f"\n{'='*60}")
    print("  Final Evaluation (Best Checkpoint)")
    print(f"{'='*60}\n")

    os.makedirs("results", exist_ok=True)

    # Load and evaluate best checkpoint.
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(best_checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)

    val_loss, top1, top5, preds, targets = evaluate(model, val_loader, cfg.device)

    print(f"  Top-1 Accuracy: {top1:.2f}%")
    print(f"  Top-5 Accuracy: {top5:.2f}%")
    print(f"  Validation Loss: {val_loss:.4f}")

    # ---- Confusion matrix ----
    cm = compute_confusion_matrix(preds, targets, cfg.num_classes)

    # Get class names
    if cfg.dataset == "cifar10":
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    else:
        class_names = None  # too many for CIFAR-100 axis labels

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        image_path=os.path.join(RESULT_DIR, "confusion_matrix.png"),
    )

    # ---- Save metrics ----
    save_metrics(
        top1, top5, cfg.dataset,
        extra=(
            f"Best Validation Accuracy (during training): {best_val_acc:.2f}%\n"
            f"Final Validation Loss: {val_loss:.4f}\n"
            f"Total Epochs: {total_epochs}\n"
            f"Batch Size: {cfg.batch_size}\n"
            f"Optimizer: AdamW (lr={adamw_lr}, weight_decay=0.05)\n"
            f"Scheduler: OneCycleLR (max_lr={adamw_lr})\n"
            f"EMA Decay: {cfg.ema_decay}\n"
            f"Mixed Precision (CUDA AMP): {use_amp}\n"
        ),
    )

    with open(final_summary_path, "w") as f:
        f.write(f"Final Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Total Epochs: {total_epochs}\n")
        f.write("Optimizer: AdamW\n")
        f.write("Scheduler: OneCycleLR\n")

    # ---- Plots ----
    save_training_plots(train_losses, val_losses, train_accs, val_accs, lr_list)

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val acc: {best_val_acc:.2f}%")
    print(f"  Final Top-1 Accuracy: {top1:.2f}%")
    print(f"  Final Top-5 Accuracy: {top5:.2f}%")
    print(f"  Saved models: {CHECKPOINT_DIR}")
    print(f"  Saved plots: {PLOT_DIR}")
    print(f"  Saved results: {RESULT_DIR}")
    print(f"{'='*60}\n")


# ======================================================================== #
#  CLI entry point                                                          #
# ======================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hybrid CNN-ViT on CIFAR-10 or CIFAR-100."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to train on (default: cifar10).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    # Build config with optional CLI overrides
    overrides = {"dataset": args.dataset}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = Config(**overrides)
    main(cfg)
