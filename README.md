# Hybrid CNN + Vision Transformer — CIFAR Accuracy Study

A **PyTorch** implementation of a **Hybrid CNN-ViT** model for image
classification on CIFAR-10 and CIFAR-100, designed for academic
reproducibility and clean experimentation.

---

## Table of Contents

1. [Model Design](#model-design)
2. [Project Structure](#project-structure)
3. [Training Strategy](#training-strategy)
4. [Hyperparameters](#hyperparameters)
5. [Quick Start](#quick-start)
6. [Expected Results](#expected-results)
7. [Requirements](#requirements)

---

## Model Design

The architecture fuses a lightweight **CNN stem** with a **Vision
Transformer (ViT)** backbone:

```
Input (3 × 32 × 32)
  │
  ├─► CNN Stem
  │     Conv(3→64)  + BN + GELU
  │     Conv(64→128) + BN + GELU          ← local feature extraction
  │
  ├─► Patch Embedding
  │     Non-overlapping 4×4 patches → 64 tokens
  │     + Learnable positional embeddings
  │
  ├─► Transformer Encoder (×6)
  │     PreNorm → Multi-Head Self-Attention (8 heads)
  │     PreNorm → MLP (4× expansion, GELU)
  │     Stochastic Depth (linearly increasing)
  │
  ├─► Global Average Pooling
  │
  └─► Linear Classifier → logits
```

### Key design choices

| Decision | Rationale |
|---|---|
| **CNN stem** instead of raw patch projection | Injects a spatial inductive bias critical for small 32×32 images. |
| **PreNorm** (LayerNorm before attention/MLP) | Stabilises training and allows higher learning rates. |
| **Stochastic Depth** | Regularises deep Transformer stacks; drop rate ramps linearly from 0→0.1. |
| **Global Average Pooling** over CLS token | Simpler, comparable accuracy, and removes the need for a dedicated token. |
| **Xavier init** for linear layers, **Kaiming** for conv layers | Standard best practices for respective layer types. |

---

## Project Structure

```
Hybrid CIFAR Accuracy Study/
│
├── config.py              # Central configuration and paths
├── model.py               # Hybrid CNN-ViT architecture
├── data.py                # Data transforms and dataloaders
├── trainer.py             # Optimizer/scheduler/train loop
├── evaluator.py           # Validation/evaluation/confusion matrix logic
├── visualization.py       # Reusable plotting helpers
├── utils.py               # Seed/device/EMA/checkpoint/csv helpers
├── training.py            # Orchestration entry point (CLI + notebook wrapper)
├── notebooks/
│   └── colab_demo.ipynb   # Notebook usage demo
├── logs/
├── checkpoints/
├── plots/
└── results/
```

---

## Training Strategy

| Component | Configuration |
|---|---|
| **Optimiser** | SGD, momentum = 0.9, Nesterov = True, weight decay = 5 × 10⁻⁴ |
| **Scheduler** | OneCycleLR (max LR = 0.1, pct_start = 0.1) |
| **Mixed Precision** | `torch.cuda.amp` (FP16 on GPU, FP32 fallback on CPU) |
| **Gradient Clipping** | Max norm = 1.0 |
| **EMA** | Exponential moving average of weights, decay = 0.999 |
| **Augmentation** | RandomCrop(32, pad=4), HFlip, RandAugment(2, 9), Normalize |
| **Epochs** | 150 |
| **Batch Size** | 128 |

---

## Hyperparameters

All values are centralised in `config.py`.

| Parameter | Default |
|---|---|
| Embedding dim | 256 |
| Transformer depth | 6 |
| Attention heads | 8 |
| MLP ratio | 4.0 |
| Dropout | 0.1 |
| Stochastic depth max | 0.1 |
| CNN channels | [64, 128] |
| Patch size | 4 |
| Random seed | 42 |

To override at the command line:

```bash
python training.py --dataset cifar100 --epochs 200 --lr 0.05 --batch-size 256
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib numpy
```

### 2. Train on CIFAR-10 (CLI)

```bash
python training.py --dataset cifar10
```

### 3. Train on CIFAR-100 (CLI)

```bash
python training.py --dataset cifar100
```

### 4. Run inside notebook / Colab

```python
from training import run_experiment

run_experiment(dataset="cifar10", epochs=150, batch_size=128, seed=42)
```

### 5. Outputs

After training completes:

- **Logs** → `logs/training_log.csv`
- **Plots** → `plots/loss_curve.png`, `plots/accuracy_curve.png`, `plots/lr_curve.png`, `plots/confusion_matrix_*.png`
- **Metrics** → `results/metrics.txt`
- **Checkpoint** → `checkpoints/best_model.pth`

---

## Expected Results

Approximate validation accuracy after 150 epochs of training (single run,
seed 42).  Results are architecture-dependent and may vary ±0.5% across
hardware and driver versions.

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
|---|---|---|
| CIFAR-10 | ~93–95% | ~99.5%+ |
| CIFAR-100 | ~75–78% | ~93–95% |

> These figures are in line with the published literature for hybrid
> CNN-ViT models of comparable size trained from scratch on CIFAR.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- matplotlib
- numpy

GPU training is automatically enabled when a CUDA device is available.

---

## Reproducibility

- All random seeds (Python, NumPy, PyTorch) are fixed via `set_seed()`.
- `torch.backends.cudnn.deterministic = True` is enforced.
- The full configuration is logged alongside training metrics.

---

## Citation

If you use this code for academic work, please cite appropriately.

---

*Prepared for academic submission — March 2026.*
