# ViT LayerNorm vs. DyT
**EN-US | [PT-BR](README-PTBR.md)**

An experimental comparison of ViT (LayerNorm) and ViT (DyT) on the CIFAR-10 dataset. This project evaluates accuracy, training stability, and resource usage between the standard Vision Transformer and its variation without normalization.

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager to ensure fast and deterministic environments for all contributors.

### 1. Prerequisites (Install `uv`)
**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*Restart the terminal after installation.*

**Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup the Repository

Clone the repository and sync the dependencies. You don't need to manually create a virtual environment; `uv` handles it automatically.

```bash
git clone https://github.com/your-username/vit-layernorm-vs-dyt.git
cd vit-layernorm-vs-dyt
uv sync
```