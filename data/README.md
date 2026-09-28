# CIFAR-10 Dataset
**EN-US | [PT-BR](/data/README-PTBR.md)**

This repository contains the documentation and details of the **CIFAR-10** dataset used for training, validating, and testing Computer Vision models. CIFAR-10 is a classic dataset of color images labeled into 10 mutually exclusive classes.

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

## 4. Preprocessing and Data Augmentation
