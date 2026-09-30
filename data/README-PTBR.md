# CIFAR-10 Dataset
**[EN-US](/data/README.md) | PT-BR**

Este repositório contém a documentação e os detalhes do [dataset CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) utilizado para o treinamento, validação e teste dos modelos de Visão Computacional. O CIFAR-10 é um conjunto de dados clássico de imagens coloridas rotuladas em 10 classes mutuamente exclusivas.

## Sumário

* [1. Visão Geral do Dataset](#1-visão-geral-do-dataset)
    * [1.1. Origem dos Dados](#11-origem-dos-dados)
* [2. Classes](#2-classes)
* [3. Divisão dos Dados](#3-divisão-dos-dados-splits)
* [4. Processamento](#4-processamento)
    * [4.1. Normalização](#41-normalização)
    * [4.2. Separação dos Dados de Validação](#42-separação-dos-dados-de-validação)

## 1. Visão Geral do Dataset

O dataset consiste em 60.000 imagens coloridas com as seguintes características e atributos fundamentais:

*   **Dimensões da Imagem:** 32x32 pixels
*   **Canais de Cor:** 3 canais (RGB - Red, Green, Blue)
*   **Shape dos Dados:** 
    *   *PyTorch:* `[C, H, W]` -> `(3, 32, 32)`
    *   *Keras/TensorFlow:* `[H, W, C]` -> `(32, 32, 3)`
*   **Total de Classes:** 10
*   **Imagens por Classe:** 6.000 imagens 
*   **Total de Amostras:** 60.000 imagens

## 1.1. Origem dos Dados

Para contular restrições de taxa (*rate limiting*) dos servidores oficiais e garantir uma configuração de ambiente rápida e confiável para todos os contribuidores, o dataset CIFAR-10 é baixado diretamente de seu repositório oficial hospedado pela Universidade de Toronto.

* **URL de Origem:** `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
* **Mantenedor:** Alex Krizhevsky (Criador do dataset CIFAR-10)

## 2. Classes

As imagens estão distribuídas uniformemente entre as 10 classes abaixo. Não há sobreposição (uma imagem pertence exclusivamente a uma classe):

| ID | Classe | Descrição |
| :--- | :--- | :--- |
| 0 | **Airplane** | Aviões (comerciais, caças, etc.) |
| 1 | **Automobile** | Carros de passeio (sedans, SUVs, etc. Caminhões não entram aqui) |
| 2 | **Bird** | Pássaros diversos |
| 3 | **Cat** | Gatos |
| 4 | **Deer** | Cervos / Veados |
| 5 | **Dog** | Cachorros |
| 6 | **Frog** | Sapos / Rãs |
| 7 | **Horse** | Cavalos |
| 8 | **Ship** | Navios e barcos |
| 9 | **Truck** | Caminhões grandes |

---

## 3. Divisão dos Dados

O dataset original fornece uma divisão padrão de Treino e Teste. Para garantir uma avaliação robusta e evitar *overfitting* durante as épocas do modelo, o conjunto de treinamento é subdividido para criar o conjunto de validação.

A distribuição adotada no pipeline será:

*   **Treinamento (Train): 45.000 imagens**
    *   Usado exclusivamente para a atualização dos pesos do modelo via *backpropagation*.
*   **Validação (Validation): 5.000 imagens**
    *   Separadas a partir do conjunto de treino original.
    *   Usado para avaliar o modelo ao final de cada época, ajustar hiperparâmetros (como *learning rate*) e aplicar *Early Stopping*.
*   **Teste (Test): 10.000 imagens**
    *   O conjunto de teste oficial do CIFAR-10 (1.000 imagens por classe).
    *   Usado **apenas uma vez** ao final do projeto para reportar as métricas finais de performance (Acurácia, F1-Score, etc.).

---

## 4. Processamento

## 4.1. Normalização

As estatísticas de normalização (`mean = [0.4914, 0.4822, 0.4465]` e `std = [0.2470, 0.2435, 0.2616]`) foram obtidas diretamente a partir da análise empírica canônica do **dataset CIFAR-10**, originalmente coletado e publicado por Alex Krizhevsky, Vinod Nair e Geoffrey Hinton da Universidade de Toronto.

* **Origem e Computação:** Estas constantes exatas são padrão na literatura de visão computacional. Elas são calculadas agregando os valores dos pixels de todas as 50.000 imagens de treino do CIFAR-10 separadamente para cada canal de cor (RGB).
* **Padrão da Comunidade:** Amplamente adotadas nos tutoriais oficiais do PyTorch, implementações de referência do `torchvision` e benchmarks acadêmicos (como o repositório `pytorch-cifar` de KuangLiu e implementações populares de Vision Transformers) para garantir comparações de modelos justas, reprodutíveis e estáveis.

## 4.2. Separação dos Dados de Validação

Como o CIFAR-10 não fornece um conjunto de validação nativo, as 5.000 amostras foram extraídas do conjunto de treinamento original (50.000 imagens). Para garantir que a amostra seja representativa e evitar qualquer viés de ordenação, os índices dos dados foram embaralhados aleatoriamente antes da divisão. O processo utiliza uma semente fixa (`seed = 42`) no gerador do PyTorch, assegurando que o embaralhamento e a separação sejam 100% determinísticos e reprodutíveis por qualquer contribuidor.