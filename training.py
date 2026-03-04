"""CLI and notebook-friendly orchestration for Hybrid CIFAR experiments."""

from __future__ import annotations

import argparse
import logging

import torch

from config import Config
from data import get_dataloaders
from evaluator import evaluate_best_model
from model import build_model
from trainer import build_optimizer, build_scheduler, train
from utils import CSVLogger, EMA, detect_device, ensure_required_directories, set_seed
from visualization import save_training_plots


# =======================
# Logging Configuration
# =======================


def _configure_logging() -> None:
    """Configure project-wide logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# =======================
# Pipeline Orchestration
# =======================


def main(cfg: Config) -> dict[str, float]:
    """
    Run complete experiment pipeline.

    Parameters
    ----------
    cfg : Config
        Full experiment configuration.

    Returns
    -------
    dict[str, float]
        Final evaluation summary metrics.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)

    # =======================
    # Setup
    # =======================
    ensure_required_directories()
    set_seed(cfg.seed)
    cfg.device = detect_device()

    logger.info("Starting experiment | dataset=%s | device=%s", cfg.dataset, cfg.device)

    # =======================
    # Data
    # =======================
    train_loader, val_loader = get_dataloaders(cfg)

    # =======================
    # Model
    # =======================
    model = build_model(cfg).to(cfg.device)

    # =======================
    # Optimization
    # =======================
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, train_loader, cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.device == "cuda" and cfg.use_amp),
    )
    ema = EMA(model, decay=cfg.ema_decay)

    # =======================
    # Training Loop
    # =======================
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        cfg=cfg,
    )

    # =======================
    # Persist Epoch Metrics
    # =======================
    csv_logger = CSVLogger()
    for epoch, train_loss, train_acc, val_loss, val_acc, learning_rate in zip(
        history["epochs"],
        history["train_loss"],
        history["train_acc"],
        history["val_loss"],
        history["val_acc"],
        history["lr"],
    ):
        csv_logger.log(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.4f}",
                "learning_rate": f"{learning_rate:.8f}",
            }
        )

    # =======================
    # Evaluation
    # =======================
    summary = evaluate_best_model(model, val_loader, cfg)

    # =======================
    # Visualization
    # =======================
    save_training_plots(history)

    logger.info(
        "Training complete | best_val=%.2f%% | top1=%.2f%% | top5=%.2f%%",
        summary["best_val_acc"],
        summary["top1"],
        summary["top5"],
    )

    return summary


# =======================
# Notebook-Friendly Wrapper
# =======================


def run_experiment(
    dataset: str = "cifar10",
    epochs: int = 150,
    batch_size: int = 128,
    seed: int = 42,
) -> dict[str, float]:
    """
    Run experiment from notebook/Colab with simple arguments.

    Parameters
    ----------
    dataset : str
        Dataset name: "cifar10" or "cifar100".
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        Final evaluation metrics from main().
    """
    cfg = Config(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    return main(cfg)


# =======================
# CLI Entry Point
# =======================


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid CNN-ViT on CIFAR-10/CIFAR-100.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    config = Config(
        dataset=arguments.dataset,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )
    main(config)
