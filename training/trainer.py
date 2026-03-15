"""Shared training pipeline supporting all four experiments."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.utils import EMA, save_checkpoint


# ---------------------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, lr: float = 3e-4,
                    weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """Build AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, train_loader, epochs, lr,
                    scheduler_type="onecycle"):
    """Build learning rate scheduler."""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )
    # Default: OneCycleLR
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
    )


# ---------------------------------------------------------------------------
# Single epoch train / validate
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                ema, device, use_amp, grad_clip):
    """Train one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_device = "cuda" if device == "cuda" else "cpu"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    """Validate one epoch. Returns (avg_loss, accuracy%)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes=10):
    """Full evaluation: top-1, top-5, predictions, targets."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    top1_c = top5_c = total = 0
    all_preds, all_tgts = [], []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        top1_c += logits.argmax(1).eq(targets).sum().item()
        top5_preds = logits.topk(5, dim=1).indices
        top5_c += top5_preds.eq(targets.unsqueeze(1)).any(1).sum().item()
        total += targets.size(0)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_tgts.extend(targets.cpu().numpy())

    return (
        total_loss / total,
        100.0 * top1_c / total,
        100.0 * top5_c / total,
        np.array(all_preds),
        np.array(all_tgts),
    )


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    ema_decay: float = 0.999,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.1,
    label: str = "experiment",
    use_amp: bool = False,
    scheduler_type: str = "onecycle",
) -> Dict[str, Any]:
    """
    Train a model with the shared pipeline.

    Returns history dict with train_losses, val_losses, train_accs,
    val_accs, lrs, best_val_acc, best_state, training_time.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = build_scheduler(optimizer, train_loader, num_epochs, lr, scheduler_type)
    scaler = GradScaler("cuda", enabled=(use_amp and device == "cuda"))
    ema = EMA(model, decay=ema_decay)

    history: Dict[str, Any] = dict(
        train_losses=[], val_losses=[], train_accs=[], val_accs=[], lrs=[],
    )
    best_val_acc = 0.0
    best_state = None
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, ema, device, use_amp, grad_clip,
        )

        ema.apply(model)
        vl_loss, vl_acc = val_epoch(model, val_loader, criterion, device)
        ema.restore(model)

        cur_lr = scheduler.get_last_lr()[0]
        history["train_losses"].append(tr_loss)
        history["val_losses"].append(vl_loss)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)
        history["lrs"].append(cur_lr)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"[{label}] Ep {epoch:03d}/{num_epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.1f}%  "
                  f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.1f}%  "
                  f"lr={cur_lr:.2e}  ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    history["best_val_acc"] = best_val_acc
    history["best_state"] = best_state
    history["training_time"] = total_time
    print(f"\nBest val acc [{label}]: {best_val_acc:.2f}% (total time: {total_time:.0f}s)\n")
    return history
