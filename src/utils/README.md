# UTILS
**EN-US | [PT-BR](/src/utils/README-PTBR.md)**

This directory contains the core modular components (building blocks) for the Vision Transformer models. An explicit, manual implementation of these layers—inspired by reference libraries such as [torchvision](https://github.com/pytorch/vision/tree/main) was chosen to ensure full transparency of the tensor flow. This avoids black-box abstractions and grants absolute control over the architecture and hyperparameters during experimentation.

## Table of Contents

* [1. Classes](#1-classes)
    * [1.1. PatchEmbedding](#11-patchembedding)
    * [1.2. TransformerEncoderBlock](#12-transformerencoderblock)

## 1. Classes

### 1.1. PatchEmbedding

* **Theoretical Concept:** Proposed by Google Brain in the original ViT paper (*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, Dosovitskiy et al., 2020).
* **Convolution Optimization:** The approach of using an `nn.Conv2d` layer with `kernel_size` and `stride` equal to the patch size was not in the original paper. It is an engineering trick popularized by Ross Wightman in the [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models/tree/main) library.
* **Technical Rationale:** Mathematically, extracting a patch and passing it through a linear projection is identical to using a 2D Convolution. However, utilizing `nn.Conv2d` leverages hardware acceleration (cuDNN), extracting and projecting the patches simultaneously. This is highly more computationally efficient than manually slicing and flattening tensors.

### 1.2. TransformerEncoderBlock

* **Theoretical Concept:** The structure of Self-Attention combined with Feed-Forward Networks (MLP) is rooted in the foundational paper *Attention Is All You Need* (Vaswani et al., 2017).
* **Pre-LN (Pre-Layer Normalization) Implementation:** The structural decision to apply `nn.LayerNorm` **before** the attention mechanism and **before** the MLP is based on the paper *On Layer Normalization in the Transformer Architecture* (Xiong et al., 2020). This approach stabilizes gradients and training significantly better than the original Post-LN setup.
* **Experimental Control:** This modular approach establishes a strictly rigorous architectural baseline. It ensures that the central project experiment and adaptations (inspired by the *Transformers Without Normalization* paper) are performed cleanly, isolated, and strictly comparable, without relying on the hidden internal behaviors of native layers (e.g., `nn.TransformerEncoderLayer`).

### 1.3. DyT (Dynamic Tanh)

- **Theoretical Concept:** Proposed in the paper *Transformers without Normalization* (2025). This class replaces traditional normalization approaches (such as LayerNorm)—which rely on computationally expensive calculations of tensor statistics (mean and variance)—with a dynamically scaled Hyperbolic Tangent (Tanh) activation function.
- **Mathematical Equation:** The implementation strictly follows the paper's formulation: `DyT(x) = γ * tanh(α * x) + β`. Here, `α` (alpha) is a trainable scalar that adjusts the global scale of the input, while `γ` (gamma) and `β` (beta) are trainable channel-wise scale and shift vectors, preserving the network's affine transformation capabilities.
- **Technical Rationale:** Eliminating classical normalization reduces the computational overhead of calculating statistical moments during the forward and backward passes. The use of a dynamic `alpha` ensures the network can still stably control gradient flow from the very first epochs, effectively preventing vanishing or exploding gradients.

### 1.4. TransformerEncoderBlockDyT

- **Theoretical Concept:** Adapts the classic structure of Self-Attention and Feed-Forward Networks (MLP) to the novel normalization-free paradigm proposed in the 2025 paper.
- **Pre-DyT Implementation:** Retains the structural Pre-Norm topology (Xiong et al., 2020) but directly replaces the `nn.LayerNorm` instances with our custom `DyT` class, applying the new dynamically scaled function prior to both the attention mechanism and the MLP.
- **Experimental Control:** The logical and mathematical architecture of this block was kept strictly identical to the baseline control model (`TransformerEncoderBlock`). By altering exclusively the experiment's independent variable (the normalization/scaling mechanism), we guarantee a rigorous and fair testing environment (fair baseline) to accurately compare the direct impacts of DyT on accuracy and computational cost.