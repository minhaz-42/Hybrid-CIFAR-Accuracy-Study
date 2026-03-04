"""
config.py — Central configuration for the Hybrid CNN-ViT CIFAR Accuracy Study.

All hyper-parameters, paths, and reproducibility settings are collected here so
that every other module can import a single, authoritative ``Config`` object.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Project paths (relative to the repository root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
RESULT_DIR = os.path.join(BASE_DIR, "results")


class Config:
    """
    Master configuration dataclass (plain Python class for readability).

    Attributes
    ----------
    dataset : str
        Either ``'cifar10'`` or ``'cifar100'``.
    num_classes : int
        Derived automatically from ``dataset``.
    seed : int
        Global random seed for reproducibility.

    # --- Data ---
    batch_size : int
    num_workers : int

    # --- CNN stem ---
    cnn_channels : list[int]
        Channel sizes for the two convolutional layers in the stem.

    # --- Patch embedding ---
    patch_size : int

    # --- Transformer ---
    embed_dim : int
    depth : int
        Number of Transformer encoder layers.
    num_heads : int
    mlp_ratio : float
        Hidden-dim expansion factor inside each MLP block.
    drop_rate : float
        Dropout rate shared across attention and MLP.
    stochastic_depth_rate : float
        Maximum drop-path probability (linearly ramped per layer).

    # --- Training ---
    epochs : int
    lr : float
    momentum : float
    weight_decay : float
    nesterov : bool
    grad_clip_norm : float
    ema_decay : float
        Exponential-moving-average decay for model weights.
    use_amp : bool
        Whether to use automatic mixed-precision (AMP).

    # --- Scheduler ---
    onecycle_max_lr : float
    onecycle_pct_start : float
    onecycle_div_factor : float
    onecycle_final_div_factor : float
    """

    # ----- dataset --------------------------------------------------------
    dataset: str = "cifar10"  # "cifar10" or "cifar100"

    @property
    def num_classes(self) -> int:
        """Return number of target classes based on the chosen dataset."""
        return 10 if self.dataset == "cifar10" else 100

    seed: int = 42

    # ----- data -----------------------------------------------------------
    batch_size: int = 128
    num_workers: int = 2

    # ----- CNN stem -------------------------------------------------------
    cnn_channels: list = None  # set in __init__

    # ----- patch embedding ------------------------------------------------
    patch_size: int = 4

    # ----- transformer ----------------------------------------------------
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    stochastic_depth_rate: float = 0.1

    # ----- training -------------------------------------------------------
    epochs: int = 150
    lr: float = 3e-4
    momentum: float = 0.9
    weight_decay: float = 0.05
    nesterov: bool = True
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    use_amp: bool = True

    # ----- OneCycleLR scheduler -------------------------------------------
    onecycle_max_lr: float = 3e-4
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4

    # ----- device ---------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, **kwargs):
        """Initialise config, optionally overriding defaults via kwargs."""
        self.cnn_channels = [64, 128]
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        lines = [f"  {k}={v!r}" for k, v in sorted(attrs.items())]
        return "Config(\n" + ",\n".join(lines) + "\n)"


# ---------------------------------------------------------------------------
# Convenience: default configs for CIFAR-10 and CIFAR-100
# ---------------------------------------------------------------------------
CIFAR10_CONFIG = Config(dataset="cifar10")
CIFAR100_CONFIG = Config(dataset="cifar100")
