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