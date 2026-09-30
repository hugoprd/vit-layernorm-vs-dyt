# CIFAR-10 Dataset
**EN-US | [PT-BR](/data/README-PTBR.md)**

This repository contains the documentation and details of the [CIFAR-10 dataset](https://cave.cs.toronto.edu/kriz/cifar.html) used for training, validating, and testing Computer Vision models. CIFAR-10 is a classic dataset of color images labeled into 10 mutually exclusive classes.

## 1. Dataset Overview

The dataset consists of 60,000 color images with the following fundamental characteristics and attributes:

*   **Image Dimensions:** 32x32 pixels
*   **Color Channels:** 3 channels (RGB - Red, Green, Blue)
*   **Data Shape:**
    *   *PyTorch:* `[C, H, W]` -> `(3, 32, 32)`
    *   *Keras/TensorFlow:* `[H, W, C]` -> `(32, 32, 3)`
*   **Total Classes:** 10
*   **Images per Class:** 6,000 images
*   **Total Samples:** 60,000 images

### 1.1. Data Source

To bypass official throttling and ensure fast, reliable environment setup for all contributors, the CIFAR-10 dataset is fetched directly from its official archive hosted by the University of Toronto.

* **Source URL:** `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
* **Maintainer:** Alex Krizhevsky (Creator of the CIFAR-10 dataset)

## 2. Classes

The images are uniformly distributed among the 10 classes below. There is no overlap (an image belongs exclusively to one class):

| ID | Class | Description |
| :--- | :--- | :--- |
| 0 | **Airplane** | Airplanes (commercial, fighter jets, etc.) |
| 1 | **Automobile** | Passenger cars (sedans, SUVs, etc. Trucks are not included here) |
| 2 | **Bird** | Various birds |
| 3 | **Cat** | Cats |
| 4 | **Deer** | Deer |
| 5 | **Dog** | Dogs |
| 6 | **Frog** | Frogs / Toads |
| 7 | **Horse** | Horses |
| 8 | **Ship** | Ships and boats |
| 9 | **Truck** | Large trucks |

---

## 3. Data Splits

The original dataset provides a standard Train and Test split. To ensure robust evaluation and prevent *overfitting* during model epochs, the training set is subdivided to create a validation set.

The distribution adopted in the pipeline will be:

*   **Training (Train): 45,000 images**
    *   Used exclusively for updating model weights via *backpropagation*.
*   **Validation (Validation): 5,000 images**
    *   Separated from the original training set.
    *   Used to evaluate the model at the end of each epoch, tune hyperparameters (such as *learning rate*), and apply *Early Stopping*.
*   **Test (Test): 10,000 images**
    *   The official CIFAR-10 test set (1,000 images per class).
    *   Used **only once** at the end of the project to report final performance metrics (Accuracy, F1-Score, etc.).

---

## 4. Processing and Data Augmentation

### 4.1. Normalization

The normalization statistics (`mean = [0.4914, 0.4822, 0.4465]` and `std = [0.2470, 0.2435, 0.2616]`) are derived directly from the canonical empirical analysis of the **CIFAR-10 dataset**, originally collected and published by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton from the University of Toronto.

* **Origin & Computation:** These exact constants are standard across computer vision literature. They are calculated by aggregating the pixel values across the entire 50,000 training images of CIFAR-10 per color channel (RGB).
* **Community Standard:** Widely adopted in official PyTorch tutorials, torchvision reference implementations, and academic benchmarks (such as KuangLiu's `pytorch-cifar` repository and mainstream vision transformer implementations) to ensure fair, reproducible, and stable model comparisons.

## 4.2. Validation Data Split

Since CIFAR-10 does not provide a native validation set, 5,000 samples were extracted from the original training set (50,000 images). To ensure the sample is representative and to prevent any ordering bias, the data indices were randomly shuffled prior to the split. This process applies a fixed seed (`seed = 42`) to the PyTorch generator, guaranteeing that the shuffling and splitting are 100% deterministic and reproducible by any contributor.