# UTILS
**[EN-US](/src/utils/README.md) | PT-BR**

Este diretório contém os componentes modulares fundamentais (blocos de construção) para os modelos Vision Transformer. Optou-se pela implementação manual e explícita destas camadas, inspiradas nas bibliotecas de referência, como o [torchvision](https://github.com/pytorch/vision/tree/main) — para garantir total transparência do fluxo de tensores, evitando abstrações de "caixa-preta" e permitindo controle absoluto sobre a arquitetura e os hiperparâmetros durante os experimentos.

## Sumário

* [1. Classes](#1-classes)
    * [1.1. PatchEmbedding](#11-patchembedding)
    * [1.2. TransformerEncoderBlock](#12-transformerencoderblock)

## 1. Classes

### 1.1. PatchEmbedding

* **Conceito Teórico:** Proposto pelo Google Brain no artigo original do ViT (*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, Dosovitskiy et al., 2020).
* **Otimização via Convolução:** A ideia de usar uma camada `nn.Conv2d` com *kernel* e *stride* iguais ao tamanho do patch não estava no artigo original, sendo um truque de engenharia popularizado por Ross Wightman na biblioteca [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models/tree/main). 
* **Justificativa Técnica:** Matematicamente, extrair um patch e passá-lo por uma projeção linear é idêntico a usar uma Convolução 2D. No entanto, o uso da `nn.Conv2d` tira proveito da aceleração de hardware (cuDNN) da GPU, extraindo e projetando os patches simultaneamente de forma computacionalmente mais eficiente do que fatiar e achatar os tensores manualmente.

### 1.2. TransformerEncoderBlock

* **Conceito Teórico:** A estrutura de Atenção (*Self-Attention*) combinada com redes *Feed Forward* (MLP) baseia-se no artigo fundacional *Attention Is All You Need* (Vaswani et al., 2017).
* **Implementação Pre-LN (Pre-Layer Normalization):** A decisão estrutural de aplicar o `nn.LayerNorm` **antes** da atenção e **antes** do MLP baseia-se no artigo *On Layer Normalization in the Transformer Architecture* (Xiong et al., 2020). Esta abordagem provou estabilizar os gradientes e o treinamento significativamente melhor do que o Post-LN original.
* **Controle Experimental:** Esta abordagem modular estabelece um *baseline* arquitetural totalmente rigoroso. Isso garante que as adaptações e o experimento central do projeto (inspirado no artigo *Transformers Without Normalization*) sejam feitos de forma limpa, isolada e estritamente comparável, sem depender de comportamentos ocultos de camadas nativas (`nn.TransformerEncoderLayer`).