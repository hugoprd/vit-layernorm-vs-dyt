# ViT LayerNorm vs. DyT
**EN-US | [PT-BR](README-PTBR.md)**

An experimental comparison of ViT (LayerNorm) and ViT (DyT) on the CIFAR-10 dataset. This project evaluates accuracy, training stability, and resource usage between the standard Vision Transformer and its variation without normalization.

## Table of Contents

* [1. Getting Started](#1-getting-started)
    * [1.1. Prerequisites](#11-prerequisites-install-uv)
    * [1.2. Setting Up the Repository](#12-setting-up-the-repository)
* [2. Data Pipeline](#2-data-pipeline)

## 1. Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager to ensure fast and deterministic environments for all contributors.

### 1.1. Prerequisites
**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*Restart the terminal after installation.*

**Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2. Setup the Repository

Clone the repository and sync the dependencies. You don't need to manually create a virtual environment; `uv` handles it automatically.

```bash
git clone https://github.com/your-username/vit-layernorm-vs-dyt.git
cd vit-layernorm-vs-dyt
uv sync
```

## 2. Data Pipeline

The project utilizes the **CIFAR-10** dataset (60,000 color images of 32x32 pixels, evenly distributed across 10 classes). To ensure a rigorous experiment and a strictly comparable baseline, the data undergoes a standardized processing pipeline:

*   **Data Splits:** 45,000 images for Training, 5,000 for Validation (deterministically extracted from the original training set using `seed=42`), and 10,000 for the official Test.
*   **Normalization:** Application of the canonical empirical CIFAR-10 statistics (`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`) across all subsets to optimize gradient stability.
*   **Data Augmentation:** Dynamic application of *Random Crop* and *Horizontal Flip* exclusively on the training set.

For in-depth details regarding data origin, class distribution, methodological rationale for normalization, and pipeline script execution, please refer to the **[Comprehensive Data Documentation](data/README.md)**.