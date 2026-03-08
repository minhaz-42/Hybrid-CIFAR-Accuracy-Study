"""
generate_plots.py — Publication-quality plots for the CIFAR-10 Hybrid CNN-ViT study.

Produces 8 high-quality figures from the training log and confusion matrix:

    1. Training & Validation Accuracy (dual-axis with smoothed + raw)
    2. Training & Validation Loss (dual-axis with smoothed + raw)
    3. Learning Rate Schedule
    4. Train-Val Accuracy Gap (Generalization Gap)
    5. Confusion Matrix Heatmap (annotated)
    6. Per-Class Accuracy Bar Chart
    7. Combined Dashboard (2×2 grid: accuracy, loss, LR, gap)
    8. Training Phases Timeline

All figures saved to plots/ at 300 DPI.

Usage:
    python generate_plots.py
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(BASE_DIR, "logs", "training_log.csv")
CM_CSV = os.path.join(BASE_DIR, "results", "confusion_matrix.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

# Color palette (professional — inspired by seaborn/muted)
C_TRAIN = "#2196F3"       # blue
C_VAL = "#FF5722"         # deep orange
C_LR = "#9C27B0"          # purple
C_GAP = "#4CAF50"         # green
C_BEST = "#E91E63"        # pink accent
C_BG = "#FAFAFA"          # light background
C_GRID = "#E0E0E0"        # grid

DPI = 300

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_training_log() -> dict:
    """Parse training_log.csv and return arrays."""
    epochs, train_loss, train_acc, val_loss, val_acc, lr = [], [], [], [], [], []

    with open(LOG_CSV, "r") as f:
        reader = csv.DictReader(f)
        seen = {}
        for row in reader:
            ep = int(row["epoch"])
            seen[ep] = row  # keep last occurrence of each epoch

    for ep in sorted(seen.keys()):
        row = seen[ep]
        epochs.append(ep)
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_accuracy"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_accuracy"]))
        lr.append(float(row["learning_rate"]))

    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(train_loss),
        "train_acc": np.array(train_acc),
        "val_loss": np.array(val_loss),
        "val_acc": np.array(val_acc),
        "lr": np.array(lr),
    }


def load_confusion_matrix() -> np.ndarray:
    return np.loadtxt(CM_CSV, delimiter=",", dtype=np.int64)


def style_axis(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, color=C_GRID, linewidth=0.5)
    ax.set_facecolor(C_BG)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_zigzag_series(ax, x, y, color, label, sparse_threshold=60):
    """Draw a series with marker-forward styling so zigzags remain visible."""
    is_sparse = len(x) <= sparse_threshold
    marker = "o" if is_sparse else None
    marker_size = 4.0 if is_sparse else 0
    line_width = 1.6 if is_sparse else 1.8
    ax.plot(
        x,
        y,
        color=color,
        linewidth=line_width,
        marker=marker,
        markersize=marker_size,
        markerfacecolor="white" if is_sparse else color,
        markeredgewidth=1.0 if is_sparse else 0,
        label=label,
    )


# ---------------------------------------------------------------------------
# Plot 1: Training & Validation Accuracy
# ---------------------------------------------------------------------------

def plot_accuracy(data: dict):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    ep = data["epochs"]
    plot_zigzag_series(ax, ep, data["train_acc"], C_TRAIN, "Train Accuracy")
    plot_zigzag_series(ax, ep, data["val_acc"], C_VAL, "Val Accuracy")

    # Mark best epoch
    best_idx = np.argmax(data["val_acc"])
    best_ep = ep[best_idx]
    best_acc = data["val_acc"][best_idx]
    ax.axvline(best_ep, color=C_BEST, linestyle="--", alpha=0.5, linewidth=1)
    ax.scatter([best_ep], [best_acc], color=C_BEST, s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(
        f"Best: {best_acc:.2f}%\n(Ep {best_ep})",
        xy=(best_ep, best_acc), xytext=(best_ep + 8, best_acc - 4),
        fontsize=9, fontweight="bold", color=C_BEST,
        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.2),
    )

    # Training phase shading
    ax.axvspan(1, 15, alpha=0.06, color="#FFC107", label="Warmup (1–15)")
    ax.axvspan(44, 46, alpha=0.12, color="#9C27B0", label="Peak LR (44–46)")

    style_axis(ax, "CIFAR-10 — Training & Validation Accuracy", "Epoch", "Accuracy (%)")
    ax.set_ylim(15, 100)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_accuracy_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [1/8] Accuracy curves saved.")


# ---------------------------------------------------------------------------
# Plot 2: Training & Validation Loss
# ---------------------------------------------------------------------------

def plot_loss(data: dict):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    ep = data["epochs"]
    plot_zigzag_series(ax, ep, data["train_loss"], C_TRAIN, "Train Loss")
    plot_zigzag_series(ax, ep, data["val_loss"], C_VAL, "Val Loss")

    # Mark minimum val loss
    min_idx = np.argmin(data["val_loss"])
    min_ep = ep[min_idx]
    min_loss = data["val_loss"][min_idx]
    ax.scatter([min_ep], [min_loss], color=C_BEST, s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(
        f"Min Val Loss: {min_loss:.4f}\n(Ep {min_ep})",
        xy=(min_ep, min_loss), xytext=(min_ep + 10, min_loss + 0.15),
        fontsize=9, fontweight="bold", color=C_BEST,
        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.2),
    )

    style_axis(ax, "CIFAR-10 — Training & Validation Loss", "Epoch", "Loss")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_loss_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] Loss curves saved.")


# ---------------------------------------------------------------------------
# Plot 3: Learning Rate Schedule
# ---------------------------------------------------------------------------

def plot_lr_schedule(data: dict):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")

    ep = data["epochs"]
    ax.plot(ep, data["lr"] * 1000, color=C_LR, linewidth=2.0)
    ax.fill_between(ep, 0, data["lr"] * 1000, alpha=0.15, color=C_LR)

    # Annotate peak
    peak_idx = np.argmax(data["lr"])
    peak_ep = ep[peak_idx]
    peak_lr = data["lr"][peak_idx]
    ax.scatter([peak_ep], [peak_lr * 1000], color=C_LR, s=60, zorder=5, edgecolors="white")
    ax.annotate(
        f"Peak: {peak_lr:.6f}\n(Ep {peak_ep})",
        xy=(peak_ep, peak_lr * 1000), xytext=(peak_ep + 15, peak_lr * 1000 * 0.85),
        fontsize=9, color=C_LR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LR, lw=1),
    )

    style_axis(ax, "CIFAR-10 — OneCycleLR Schedule", "Epoch", "Learning Rate (×10⁻³)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_learning_rate_schedule.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [3/8] LR schedule saved.")


# ---------------------------------------------------------------------------
# Plot 4: Generalization Gap
# ---------------------------------------------------------------------------

def plot_generalization_gap(data: dict):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    ep = data["epochs"]
    gap = data["train_acc"] - data["val_acc"]
    ax.fill_between(ep, 0, gap, alpha=0.15, color=C_GAP)
    ax.plot(ep, gap, color=C_GAP, linewidth=1.8, label="Train-Val Gap")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Annotate final gap
    final_gap = gap[-1]
    ax.annotate(
        f"Final gap: {final_gap:.1f}%",
        xy=(ep[-1], final_gap), xytext=(ep[-1] - 30, final_gap + 1),
        fontsize=9, fontweight="bold", color=C_GAP,
        arrowprops=dict(arrowstyle="->", color=C_GAP, lw=1),
    )

    style_axis(ax, "CIFAR-10 — Generalization Gap (Train Acc − Val Acc)", "Epoch", "Accuracy Gap (%)")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_generalization_gap.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [4/8] Generalization gap saved.")


# ---------------------------------------------------------------------------
# Plot 5: Confusion Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")

    # Normalize for color mapping (row-wise = per-class recall %)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#BBDEFB", "#2196F3", "#0D47A1"])
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)

    # Add color bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=10)

    # Annotate cells with counts
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = "white" if pct > 60 else "black"
            fontsize = 8 if val < 100 else 7
            text = f"{val}\n({pct:.0f}%)" if i == j else f"{val}"
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("CIFAR-10 — Confusion Matrix (Best Checkpoint)", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_confusion_matrix.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [5/8] Confusion matrix saved.")


# ---------------------------------------------------------------------------
# Plot 6: Per-Class Accuracy Bar Chart
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    n = cm.shape[0]
    per_class_acc = np.diag(cm) / cm.sum(axis=1) * 100
    overall = np.trace(cm) / cm.sum() * 100

    # Sort by accuracy for better visualization
    sorted_idx = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[sorted_idx]
    sorted_names = [CIFAR10_CLASSES[i] for i in sorted_idx]

    # Color gradient based on accuracy
    colors = []
    for acc in sorted_acc:
        if acc >= 95:
            colors.append("#4CAF50")  # green
        elif acc >= 90:
            colors.append("#2196F3")  # blue
        elif acc >= 85:
            colors.append("#FF9800")  # orange
        else:
            colors.append("#F44336")  # red

    bars = ax.barh(range(n), sorted_acc, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Add value labels
    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")

    # Overall accuracy line
    ax.axvline(overall, color=C_BEST, linestyle="--", linewidth=1.5, alpha=0.7, label=f"Overall: {overall:.1f}%")

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlim(75, 102)
    style_axis(ax, "CIFAR-10 — Per-Class Accuracy (Sorted)", "", "Accuracy (%)")
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_per_class_accuracy.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [6/8] Per-class accuracy saved.")


# ---------------------------------------------------------------------------
# Plot 7: Combined Dashboard (2×2)
# ---------------------------------------------------------------------------

def plot_dashboard(data: dict):
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.25)

    ep = data["epochs"]

    # --- Top-left: Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    plot_zigzag_series(ax1, ep, data["train_acc"], C_TRAIN, "Train")
    plot_zigzag_series(ax1, ep, data["val_acc"], C_VAL, "Validation")
    best_idx = np.argmax(data["val_acc"])
    ax1.scatter([ep[best_idx]], [data["val_acc"][best_idx]], color=C_BEST, s=50, zorder=5, edgecolors="white")
    style_axis(ax1, "Accuracy", "Epoch", "Accuracy (%)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(15, 100)

    # --- Top-right: Loss ---
    ax2 = fig.add_subplot(gs[0, 1])
    plot_zigzag_series(ax2, ep, data["train_loss"], C_TRAIN, "Train")
    plot_zigzag_series(ax2, ep, data["val_loss"], C_VAL, "Validation")
    style_axis(ax2, "Loss", "Epoch", "Loss")
    ax2.legend(fontsize=8, loc="upper right")

    # --- Bottom-left: LR ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, data["lr"] * 1000, color=C_LR, linewidth=2)
    ax3.fill_between(ep, 0, data["lr"] * 1000, alpha=0.12, color=C_LR)
    style_axis(ax3, "Learning Rate Schedule", "Epoch", "LR (×10⁻³)")

    # --- Bottom-right: Gap ---
    ax4 = fig.add_subplot(gs[1, 1])
    gap = data["train_acc"] - data["val_acc"]
    ax4.fill_between(ep, 0, gap, alpha=0.15, color=C_GAP)
    ax4.plot(ep, gap, color=C_GAP, linewidth=1.8, label="Gap")
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    style_axis(ax4, "Generalization Gap", "Epoch", "Train − Val Acc (%)")
    ax4.legend(fontsize=8)

    fig.suptitle("CIFAR-10 Hybrid CNN-ViT — Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(PLOT_DIR, "07_training_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [7/8] Training dashboard saved.")


# ---------------------------------------------------------------------------
# Plot 8: Training Phases Timeline
# ---------------------------------------------------------------------------

def plot_training_phases(data: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="white", sharex=True)

    ep = data["epochs"]

    # Define phases with colors
    phases = [
        (1, 15, "Warmup\n& Early", "#FFF9C4", "#F9A825"),
        (16, 45, "Rapid\nConvergence", "#C8E6C9", "#2E7D32"),
        (46, 85, "Refinement\n(LR Decay)", "#BBDEFB", "#1565C0"),
        (86, 121, "Fine-Tuning\n(Low LR)", "#E1BEE7", "#7B1FA2"),
        (122, 150, "Final\nPlateau", "#FFCCBC", "#D84315"),
    ]

    for start, end, label, bg_color, text_color in phases:
        ax1.axvspan(start, end, alpha=0.3, color=bg_color)
        ax2.axvspan(start, end, alpha=0.3, color=bg_color)
        mid = (start + end) / 2
        ax1.text(mid, 98, label, ha="center", va="top", fontsize=7.5,
                 fontweight="bold", color=text_color)

    # Top: Accuracy
    plot_zigzag_series(ax1, ep, data["train_acc"], C_TRAIN, "Train Acc")
    plot_zigzag_series(ax1, ep, data["val_acc"], C_VAL, "Val Acc")
    best_idx = np.argmax(data["val_acc"])
    ax1.scatter([ep[best_idx]], [data["val_acc"][best_idx]], color=C_BEST, s=60, zorder=5,
                edgecolors="white", linewidth=1.5)
    style_axis(ax1, "", "", "Accuracy (%)")
    ax1.set_title("CIFAR-10 — Training Phases & Metrics Timeline", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim(15, 100)

    # Bottom: Loss
    plot_zigzag_series(ax2, ep, data["train_loss"], C_TRAIN, "Train Loss")
    plot_zigzag_series(ax2, ep, data["val_loss"], C_VAL, "Val Loss")
    style_axis(ax2, "", "Epoch", "Loss")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "08_training_phases.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [8/8] Training phases saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Generating publication-quality plots for CIFAR-10")
    print("=" * 60)
    print()

    data = load_training_log()
    cm = load_confusion_matrix()

    print(f"  Loaded {len(data['epochs'])} epochs of training data")
    print(f"  Confusion matrix shape: {cm.shape}")
    print(f"  Output directory: {PLOT_DIR}/")
    print()

    plot_accuracy(data)
    plot_loss(data)
    plot_lr_schedule(data)
    plot_generalization_gap(data)
    plot_confusion_matrix(cm)
    plot_per_class_accuracy(cm)
    plot_dashboard(data)
    plot_training_phases(data)

    print()
    print("=" * 60)
    print("  All 8 plots saved to plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
