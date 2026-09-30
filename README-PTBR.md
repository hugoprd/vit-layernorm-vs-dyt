# ViT LayerNorm vs. DyT
**[EN-US](README.md) | PT-BR**

Uma comparação experimental entre o ViT (LayerNorm) e o ViT (DyT) no dataset CIFAR-10. Este projeto avalia a acurácia, estabilidade de treinamento e o custo computacional entre o Vision Transformer padrão e sua variação sem normalização.

O projeto foi desenvolvido como trabalho de conclusão da disciplina Aprendizagem Profunda (AP) da Universidade Federal do Estado do Rio de Janeiro (UNIRIO) e teve como base fundamental o artigo **Transformers without Normalization** de 2025.

## Sumário

* [1. Começando](#1-começando)
    * [1.1. Pré-requisitos](#11-pré-requisitos-instalar-o-uv)
    * [1.2. Configurando o Repositório](#12-configurando-o-repositório)
* [2. Pipeline de Dados](#2-pipeline-de-dados)

## 1. Começando

Este projeto utiliza o [`uv`](https://github.com/astral-sh/uv) como gerenciador de pacotes para garantir um ambiente rápido, padronizado e determinístico para todos os contribuidores.

### 1.1. Pré-requisitos
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

## 2. Pipeline de Dados

O projeto utiliza o dataset **CIFAR-10** (60.000 imagens coloridas de 32x32 pixels, distribuídas em 10 classes). Para garantir um experimento rigoroso e um *baseline* comparável, os dados passam por um pipeline de processamento padronizado:

*   **Divisão (Splits):** 45.000 imagens para Treinamento, 5.000 para Validação (separadas de forma determinística utilizando `seed=42`) e 10.000 para Teste oficial.
*   **Normalização:** Aplicação das estatísticas empíricas canônicas do CIFAR-10 (`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`) em todos os subconjuntos para otimizar a estabilidade do gradiente.
*   **Data Augmentation:** Aplicação dinâmica de *Random Crop* e *Horizontal Flip* exclusivamente no conjunto de treinamento.

Para detalhes aprofundados sobre a origem dos dados, distribuição de classes, justificativa metodológica da normalização e execução dos scripts de pipeline, consulte a **[Documentação de Dados Completa](data/README-PTBR.md)**.