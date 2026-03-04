"""
model.py — Hybrid CNN + Vision Transformer for CIFAR-10 / CIFAR-100.

Architecture overview
---------------------
1. **CNN Stem** — two convolution layers with BatchNorm and GELU that
   progressively expand the channel dimension while preserving spatial
   resolution.  This gives the Transformer a richer, locally-aware input
   representation compared to a naive linear patch projection.

2. **Patch Embedding** — after the CNN stem the feature map is reshaped
   into a sequence of non-overlapping patches (patch_size × patch_size)
   and linearly projected to ``embed_dim``.  Learnable positional
   embeddings are added.

3. **Transformer Encoder** — a stack of ``depth`` PreNorm encoder blocks
   each consisting of Multi-Head Self-Attention and a two-layer MLP with
   GELU activation.  Stochastic Depth (drop-path) is applied with a
   linearly increasing probability across layers.

4. **Classification Head** — Global Average Pooling over the sequence
   dimension followed by LayerNorm and a linear classifier.  Using GAP
   instead of a CLS token yields comparable accuracy while being simpler.

Design rationale
~~~~~~~~~~~~~~~~
* **PreNorm** (LayerNorm *before* attention / MLP) stabilises training
  and allows higher learning rates, which is critical when training from
  scratch on small-scale datasets like CIFAR.
* **Stochastic Depth** acts as a strong regulariser and improves
  generalisation for deeper Transformer stacks.
* **CNN stem** injects an inductive bias for local patterns that pure
  ViTs struggle to learn from 32×32 images alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================== #
#  Building blocks                                                          #
# ======================================================================== #

class DropPath(nn.Module):
    """
    Stochastic Depth (drop-path) regularisation.

    During training, each sample's residual branch is randomly dropped
    with probability ``drop_prob``.  At test time nothing is dropped.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (B, 1, 1, ...) — broadcastable
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class PreNorm(nn.Module):
    """Apply LayerNorm *before* a sub-layer (Pre-Norm architecture)."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


# ======================================================================== #
#  Multi-Head Self-Attention                                                #
# ======================================================================== #

class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Parameters
    ----------
    dim : int
        Input / output dimensionality.
    num_heads : int
        Number of attention heads.
    drop_rate : float
        Dropout applied to the attention weights.
    """

    def __init__(self, dim: int, num_heads: int = 8, drop_rate: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project and reshape: (B, N, 3, H, D) -> (3, B, H, N, D)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


# ======================================================================== #
#  Feed-Forward Network (MLP)                                               #
# ======================================================================== #

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation used inside each Transformer block.

    Parameters
    ----------
    dim : int
        Input / output dimensionality.
    hidden_dim : int
        Hidden layer width (typically ``dim * mlp_ratio``).
    drop_rate : float
        Dropout applied after each linear layer.
    """

    def __init__(self, dim: int, hidden_dim: int, drop_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================== #
#  Transformer Encoder Block                                                #
# ======================================================================== #

class TransformerBlock(nn.Module):
    """
    A single Transformer encoder layer with PreNorm and Stochastic Depth.

    Residual connections wrap both the attention and MLP sub-layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadAttention(dim, num_heads, drop_rate))
        self.ff = PreNorm(
            dim,
            FeedForward(dim, int(dim * mlp_ratio), drop_rate),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.ff(x))
        return x


# ======================================================================== #
#  CNN Stem                                                                 #
# ======================================================================== #

class CNNStem(nn.Module):
    """
    Lightweight CNN feature extractor placed before the Transformer.

    Two convolutional stages with BatchNorm and GELU inject a local
    inductive bias that significantly improves ViT training on small
    images (32×32).  Spatial resolution is preserved (``padding=1``).

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB).
    channels : list[int]
        Channel sizes for the two conv layers, e.g. ``[64, 128]``.
    """

    def __init__(self, in_channels: int = 3, channels: list[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [64, 128]
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 3, H, W) -> (B, C_out, H, W)."""
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return x


# ======================================================================== #
#  Patch Embedding                                                          #
# ======================================================================== #

class PatchEmbedding(nn.Module):
    """
    Convert a 2-D feature map into a sequence of patch embeddings.

    Uses a non-overlapping convolution with ``kernel_size = stride =
    patch_size`` to avoid explicit reshape gymnastics.  Learnable
    positional embeddings are added to the resulting sequence.

    Parameters
    ----------
    in_channels : int
        Number of channels coming out of the CNN stem.
    embed_dim : int
        Transformer embedding dimensionality.
    patch_size : int
        Side length of each square patch.
    img_size : int
        Spatial size of the input feature map (assumes square).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        img_size: int = 32,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, embed_dim)."""
        x = self.proj(x)                           # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)           # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x


# ======================================================================== #
#  Full Hybrid CNN-ViT Model                                                #
# ======================================================================== #

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Vision Transformer for CIFAR classification.

    Forward flow::

        image -> CNN Stem -> Patch Embed -> Transformer Encoder -> GAP -> MLP Head

    Parameters
    ----------
    img_size : int
        Spatial size of the input image (32 for CIFAR).
    in_channels : int
        Number of input channels (3 for RGB).
    num_classes : int
        Number of output classes (10 or 100).
    cnn_channels : list[int]
        Channel sizes for the CNN stem.
    patch_size : int
        Patch size used in patch embedding.
    embed_dim : int
        Transformer hidden dimensionality.
    depth : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads per layer.
    mlp_ratio : float
        Expansion ratio for the MLP inside each Transformer block.
    drop_rate : float
        Dropout rate for attention and MLP layers.
    stochastic_depth_rate : float
        Maximum stochastic depth drop probability (ramped linearly).
    """

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        cnn_channels: list[int] | None = None,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        stochastic_depth_rate: float = 0.1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128]

        # --- CNN feature extractor ---
        self.stem = CNNStem(in_channels, cnn_channels)

        # --- Patch embedding (from CNN output) ---
        self.patch_embed = PatchEmbedding(
            in_channels=cnn_channels[-1],
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

        # --- Transformer encoder stack ---
        # Linearly increasing drop-path rate
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # --- Classification head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight initialisation                                               #
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """
        Apply Xavier-uniform to linear layers and small-constant init
        for LayerNorm, following ViT best practices.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.stem(x)                   # (B, C_stem, 32, 32)
        x = self.patch_embed(x)            # (B, num_patches, embed_dim)
        x = self.blocks(x)                 # (B, num_patches, embed_dim)
        x = self.norm(x)                   # (B, num_patches, embed_dim)
        x = x.mean(dim=1)                  # Global Average Pooling -> (B, embed_dim)
        x = self.head(x)                   # (B, num_classes)
        return x


# ======================================================================== #
#  Factory helper                                                           #
# ======================================================================== #

def build_model(cfg) -> HybridCNNViT:
    """
    Instantiate a ``HybridCNNViT`` model from a ``Config`` object.

    Parameters
    ----------
    cfg : config.Config
        Configuration object containing all architecture hyper-parameters.

    Returns
    -------
    HybridCNNViT
        The constructed model, ready to be moved to ``cfg.device``.
    """
    return HybridCNNViT(
        img_size=32,
        in_channels=3,
        num_classes=cfg.num_classes,
        cnn_channels=cfg.cnn_channels,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_rate=cfg.drop_rate,
        stochastic_depth_rate=cfg.stochastic_depth_rate,
    )
