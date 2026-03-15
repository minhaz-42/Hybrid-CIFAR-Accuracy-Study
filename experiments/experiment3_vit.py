"""Experiment 3 — Vision Transformer (ViT).

Run:  python -m experiments.experiment3_vit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.cifar_loader import get_dataloaders
from models.vision_transformer import VisionTransformer
from training.trainer import run_experiment
from training.utils import set_seed, detect_device, ensure_dirs, count_parameters


def main():
    set_seed(42)
    ensure_dirs()
    device = detect_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=128)

    model = VisionTransformer(
        img_size=32,
        in_channels=3,
        num_classes=10,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.1,
        stochastic_depth_rate=0.1,
    )
    print(f"Vision Transformer — trainable parameters: {count_parameters(model):,}")

    history = run_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        lr=3e-4,
        weight_decay=0.05,
        ema_decay=0.999,
        grad_clip=1.0,
        label_smoothing=0.1,
        label="Exp-3 ViT",
        use_amp=(device == "cuda"),
    )
    return history


if __name__ == "__main__":
    main()
