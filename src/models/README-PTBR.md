# MODELS
**[EN-US](/src/models/README.md) | PT-BR**

## Sumário

* [1. Hiperparâmetros](#1-hiperparâmetros)
    * [1.1. Justificativa dos Hiperparâmetros](#11-justificativa-dos-hiperparâmetros)

## 1. Hiperparâmetros

A tabela abaixo detalha a configuração base utilizada na instanciação dos modelos para o experimento no dataset CIFAR-10:

| Parâmetro | Valor | Descrição |
| :--- | :--- | :--- |
| `img_channels` | 3 | Canais de cor das imagens de entrada (RGB). |
| `patch_size` | 4 | Tamanho da grade de cada patch extraído (4x4 pixels). |
| `embed_dim` | 256 | Dimensão dos embeddings (tamanho do espaço latente). |
| `depth` | 6 | Número de blocos do Transformer (Encoder). |
| `num_heads` | 8 | Número de cabeças de atenção (*Multi-Head Attention*). |
| `mlp_ratio` | 4.0 | Fator de expansão da rede *Feed-Forward* (MLP). |
| `num_classes` | 10 | Quantidade de classes de saída (CIFAR-10). |
| `dropout` | 0.1 | Taxa de *dropout* utilizada para regularização. |

### 1.1. Comparação com a Arquitetura Original

Para adaptar as propostas do artigo base à realidade do nosso experimento, os hiperparâmetros originais sofreram uma redução de escala (*scaling down*). A tabela a seguir mapeia a diferença entre a configuração oficial do modelo ViT-Base (idealizado para o ImageNet) e a configuração adotada neste projeto para o CIFAR-10:

| Parâmetro | ViT-Base Original (Artigo) | Nossa Configuração (CIFAR-10) |
| :--- | :--- | :--- |
| `patch_size` | 16 | 4 |
| `embed_dim` | 768 | 256 |
| `depth` | 12 | 6 |
| `num_heads` | 12 | 8 |
| `mlp_ratio` | 4.0 | 4.0 |

### 1.3. Justificativa das Adaptações Arquiteturais

A redução de escala dos hiperparâmetros não foi feita de forma arbitrária, mas sim orientada pelas restrições dimensionais do dataset e pelos princípios teóricos do ViT-Lite:

* **Tamanho do Patch (`patch_size=4`):** O artigo original utiliza imagens de alta resolução (224x224) com patches de 16x16. Como o dataset CIFAR-10 possui imagens de apenas 32x32 pixels, aplicar um `patch_size=16` geraria uma grade de apenas 4 tokens (2x2), inviabilizando a eficácia do mecanismo de *Self-Attention*. Reduzimos o tamanho para 4x4, garantindo uma sequência de 64 tokens expressivos por imagem.
* **Profundidade e Dimensão Latente (`depth=6`, `embed_dim=256`):** Modelos de linguagem e visão originais empregam redes extremamente largas e profundas devido à abundância de dados (ex: ImageNet com milhões de imagens). Instanciar 12 blocos de atenção e 768 dimensões com apenas 45.000 amostras causaria *overfitting* imediato e memorização do conjunto de treino. Cortar esses valores pela metade força o modelo a aprender representações mais generalistas (seguindo a literatura de ViTs para pequenos datasets).
* **Cabeças de Atenção (`num_heads=8`):** O número de cabeças foi reduzido de 12 para 8 proporcionalmente à diminuição do tamanho do *embedding*. Com `embed_dim=256`, cada cabeça processa 32 dimensões (`256 / 8 = 32`). Isso mantém uma distribuição saudável da atenção e evita a fragmentação matemática excessiva do espaço latente que ocorreria se mantivéssemos 12 cabeças.