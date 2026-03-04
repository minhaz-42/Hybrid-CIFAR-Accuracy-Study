"""Training utilities and loop orchestration."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config
from evaluator import validate
from utils import EMA, save_checkpoint


# =======================
# Logging
# =======================

logger = logging.getLogger(__name__)


# =======================
# Optimizer / Scheduler
# =======================

def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer from configuration.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    torch.optim.Optimizer
        AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    cfg: Config,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    Build OneCycleLR scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    train_loader : DataLoader
        Training loader.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    torch.optim.lr_scheduler.OneCycleLR
        Learning-rate scheduler.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.onecycle_max_lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.onecycle_pct_start,
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )


# =======================
# Training Loop
# =======================

def _print_batch_progress(
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    running_loss: float,
    running_acc: float,
) -> None:
    """Print in-place batch progress for live training feedback."""
    progress = (
        f"\rEpoch {epoch:03d}/{total_epochs:03d} "
        f"Batch {batch_idx:04d}/{total_batches:04d} "
        f"Loss {running_loss:.4f} "
        f"Acc {running_acc:.2f}%"
    )
    print(progress, end="", flush=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: GradScaler,
    ema: EMA,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Train one epoch of the model.

    Parameters
    ----------
    model : nn.Module
        Neural network to train.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : torch.optim.lr_scheduler.OneCycleLR
        Scheduler stepped every batch.
    scaler : GradScaler
        Gradient scaler for AMP.
    ema : EMA
        Exponential moving average tracker.
    device : str
        Compute device.
    use_amp : bool
        Whether AMP is enabled.
    grad_clip_norm : float
        Maximum gradient norm.
    epoch : int
        Current epoch number.
    total_epochs : int
        Total epochs.

    Returns
    -------
    tuple[float, float]
        Average loss and top-1 accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

        avg_loss = running_loss / total
        avg_acc = 100.0 * correct / total
        _print_batch_progress(
            epoch,
            total_epochs,
            batch_idx,
            total_batches,
            avg_loss,
            avg_acc,
        )

    print()
    return running_loss / total, 100.0 * correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: GradScaler,
    ema: EMA,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Run full training loop with validation and checkpointing.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training loader.
    val_loader : DataLoader
        Validation loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.OneCycleLR
        Scheduler.
    scaler : GradScaler
        AMP gradient scaler.
    ema : EMA
        EMA tracker.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    dict[str, Any]
        Training history containing losses, accuracies, and learning rates.
    """
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(cfg.use_amp and cfg.device == "cuda")

    history: Dict[str, Any] = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=cfg.device,
            use_amp=use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )

        ema.apply(model)
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.device)
        ema.restore(model)

        current_lr = scheduler.get_last_lr()[0]

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema.apply(model)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_val_acc,
                filename="best_model.pth",
            )
            ema.restore(model)

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %03d/%03d | train_loss %.4f | train_acc %.2f%% | "
            "val_loss %.4f | val_acc %.2f%% | lr %.6f | %.1fs",
            epoch,
            cfg.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            elapsed,
        )

    history["best_val_acc"] = best_val_acc

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        best_acc=best_val_acc,
        filename="last_model.pth",
    )

    return history
