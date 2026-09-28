# ViT LayerNorm vs. DyT
**[EN-US](README.md) | PT-BR**

Uma comparação experimental entre o ViT (LayerNorm) e o ViT (DyT) no dataset CIFAR-10. Este projeto avalia a acurácia, estabilidade de treinamento e o custo computacional entre o Vision Transformer padrão e sua variação sem normalização.

## 1. Começando

Este projeto utiliza o [`uv`](https://github.com/astral-sh/uv) como gerenciador de pacotes para garantir um ambiente rápido, padronizado e determinístico para todos os contribuidores.

### 1.1. Pré-requisitos (Instalar o `uv`)
**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*Reinicie o seu terminal depois da instalação.*

**Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2. Configurando o Repositório

Clone o repositório e sincronize as dependências. Não é necessário criar um ambiente virtual manualmente; o `uv` faz isso automaticamente.

```bash
git clone https://github.com/your-username/vit-layernorm-vs-dyt.git
cd vit-layernorm-vs-dyt
uv sync
```