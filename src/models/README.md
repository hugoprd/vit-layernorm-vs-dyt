# MODELS
**EN-US | [PT-BR](/src/models/README-PTBR.md)**

## Table of Contents

* [1. Hyperparameters](#1-hyperparameters)
    * [1.1 Hyperparameter Rationale](#11-hyperparameter-rationale)

## 1. Hyperparameters

The table below details the baseline configuration used when instantiating the models for the CIFAR-10 experiment:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `img_channels` | 3 | Input image color channels (RGB). |
| `patch_size` | 4 | Grid size of each extracted patch (4x4 pixels). |
| `embed_dim` | 256 | Embedding dimension (latent space size). |
| `depth` | 6 | Number of Transformer blocks (Encoder). |
| `num_heads` | 8 | Number of attention heads (*Multi-Head Attention*). |
| `mlp_ratio` | 4.0 | Expansion factor for the *Feed-Forward* network (MLP). |
| `num_classes` | 10 | Number of output classes (CIFAR-10). |
| `dropout` | 0.1 | Dropout rate used for regularization. |

### 1.2. Comparison with the Original Architecture

To adapt the base paper's proposals to the constraints of our experiment, the original hyperparameters underwent a structured scaling down. The following table maps the differences between the official ViT-Base configuration (designed for ImageNet) and the configuration adopted in this project for CIFAR-10:

| Parameter | Original ViT-Base (Paper) | Our Configuration (CIFAR-10) |
| :--- | :--- | :--- |
| `patch_size` | 16 | 4 |
| `embed_dim` | 768 | 256 |
| `depth` | 12 | 6 |
| `num_heads` | 12 | 8 |
| `mlp_ratio` | 4.0 | 4.0 |

### 1.3. Rationale for Architectural Adaptations

The scaling down of hyperparameters was not arbitrary; it was strictly guided by the dataset's dimensional constraints and the theoretical principles of the ViT-Lite architecture:

* **Patch Size (`patch_size=4`):** The original paper utilizes high-resolution images (224x224) with 16x16 patches. Since the CIFAR-10 dataset consists of only 32x32 pixel images, applying a `patch_size=16` would generate a grid of just 4 tokens (2x2), crippling the effectiveness of the *Self-Attention* mechanism. We reduced the patch size to 4x4 to ensure a rich sequence of 64 expressive tokens per image.
* **Network Depth and Latent Dimension (`depth=6`, `embed_dim=256`):** Original vision models deploy extremely deep and wide networks to digest massive datasets (e.g., ImageNet with millions of samples). Instantiating 12 attention blocks and 768 dimensions on merely 45,000 samples would cause severe overfitting and training set memorization. Halving these parameters forces the model to learn more generalized representations, adhering to standard practices for ViTs on small datasets.
* **Attention Heads (`num_heads=8`):** The number of heads was reduced from 12 to 8, in proportion to the decreased embedding size. With an `embed_dim=256`, each head processes exactly 32 dimensions (`256 / 8 = 32`). This maintains a healthy distribution of attention and prevents the excessive mathematical fragmentation of the latent space that would occur if 12 heads were maintained.