import platform
from matplotlib.colors import LinearSegmentedColormap
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

import torch

COLOR_PURPLE_DARK = "#4E31A4"
COLOR_PURPLE_LIGHT = "#D6CFF8"
COLOR_YELLOW_SECONDARY = "#D97706"

PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "PurpleCustom", ["#F3F0FE", COLOR_PURPLE_LIGHT, COLOR_PURPLE_DARK]
)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        """
        patience: how many epochs to wait without improvement before stopping.
        min_delta: minimum difference to be considered an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            return True
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

            return False


def plot_confusion_matrix(conf_matrix, classes: list[str], save_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap=PURPLE_CMAP,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Contagem"},
    )
    plt.ylabel("Rótulo Verdadeiro (True Label)", fontweight="bold")
    plt.xlabel("Rótulo Preditivo (Predicted Label)", fontweight="bold")
    plt.title("Matriz de Confusão", fontsize=14, fontweight="bold", color=COLOR_PURPLE_DARK)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(fpr, tpr, classes: list[str], save_path: Path):
    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10.colors

    for i in range(len(classes)):
        color = colors[i % len(colors)]
        plt.plot(
            fpr[i].cpu(),
            tpr[i].cpu(),
            lw=2,
            color=color,
            label=f"{classes[i]}",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        color=COLOR_PURPLE_LIGHT,
        lw=2,
        linestyle="--",
        label="Acaso (AUC = 0.5)",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontweight="bold")
    plt.title(
        "Curvas ROC (One-vs-Rest) Multiclasse",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PURPLE_DARK,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize="small", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(classes: list[str], csv_path: Path, output_dir: Path, logger):
    if not csv_path.is_file():
        logger.warning(f"Metrics CSV file not found at: {csv_path}. Skipping training curves.")

        return

    logger.info(f"Generating training metric curves from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Loss
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss",
        color=COLOR_PURPLE_DARK,
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Val Loss",
        color=COLOR_YELLOW_SECONDARY,
        linewidth=2,
        marker="s",
        markersize=3,
    )
    plt.title(
        "Train & Validation Loss per Epoch",
        fontsize=13,
        fontweight="bold",
        color=COLOR_PURPLE_DARK,
    )
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to: {loss_path}")

    # 2. F1 Macro & Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["epoch"],
        df["val_f1_macro"],
        label="Val F1 Macro",
        color=COLOR_PURPLE_DARK,
        linewidth=2,
        marker="^",
        markersize=3,
    )
    plt.plot(
        df["epoch"],
        df["val_accuracy"],
        label="Val Accuracy",
        color=COLOR_YELLOW_SECONDARY,
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.title(
        "Validation Metrics (F1 Macro & Accuracy) per Epoch",
        fontsize=13,
        fontweight="bold",
        color=COLOR_PURPLE_DARK,
    )
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    f1_path = output_dir / "f1_macro_curve.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    logger.info(f"F1 Macro / Accuracy curve saved to: {f1_path}")

    # 3. Throughput
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["epoch"],
        df["throughput_imgs_per_sec"],
        label="Throughput (img/s)",
        color=COLOR_PURPLE_DARK,
        linewidth=2,
        marker="d",
        markersize=3,
    )
    plt.title(
        "Training Throughput per Epoch", fontsize=13, fontweight="bold", color=COLOR_PURPLE_DARK
    )
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Images / Second", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    throughput_path = output_dir / "throughput_curve.png"
    plt.savefig(throughput_path, dpi=300)
    plt.close()
    logger.info(f"Throughput curve saved to: {throughput_path}")


def log_system_specs(logger):
    """
    Capture and log the hardware specifications of the machine
    where the experiment is being executed.
    """
    logger.info("=" * 60)
    logger.info("HARDWARE AND SYSTEM ESPECIFICATIONS")
    logger.info("=" * 60)

    # 1. Operational System
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    logger.info(f"OS        : {os_info}")

    # 2. CPU e RAM
    cpu_cores_physical = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)

    logger.info(
        f"CPU       : {cpu_cores_physical} Physical Cores / {cpu_cores_logical} Logical Cores"
    )
    logger.info(f"RAM Total : {ram_gb:.2f} GB")

    # 3. GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_ram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU       : {gpu_name}")
        logger.info(f"Total VRAM: {gpu_ram_gb:.2f} GB")
        logger.info(f"CUDA Ver. : {torch.version.cuda}")
    else:
        logger.info("GPU       : None CUDA GPU detected. Running on CPU.")

    logger.info("=" * 60)
