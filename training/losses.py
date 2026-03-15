"""Loss functions for experiments."""

from __future__ import annotations

import torch.nn as nn


def get_criterion(label_smoothing: float = 0.1) -> nn.Module:
    """Return CrossEntropyLoss with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
