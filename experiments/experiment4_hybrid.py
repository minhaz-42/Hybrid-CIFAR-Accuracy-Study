"""Experiment 4 — Hybrid CNN + Vision Transformer.

Run:  python -m experiments.experiment4_hybrid
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cifar_loader import get_dataloaders
from models.hybrid_cnn_vit import HybridCNNViT
from training.trainer import run_experiment
from training.utils import set_seed, detect_device, ensure_dirs, count_parameters


def main():
    set_seed(42)
    ensure_dirs()
    device = detect_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=128)

    model = HybridCNNViT(
        img_size=32,
        in_channels=3,
        num_classes=10,
        cnn_channels=[64, 128],
        patch_size=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.1,
        stochastic_depth_rate=0.1,
    )
    print(f"Hybrid CNN-ViT — trainable parameters: {count_parameters(model):,}")

    history = run_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=150,
        lr=3e-4,
        weight_decay=0.05,
        ema_decay=0.999,
        grad_clip=1.0,
        label_smoothing=0.1,
        label="Exp-4 Hybrid CNN-ViT",
        use_amp=(device == "cuda"),
    )
    return history


if __name__ == "__main__":
    main()
