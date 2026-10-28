import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn  # noqa E401
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassROC,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from logs.cmd.logger_config import setup_logger
from src.models.vit_layernorm import ViT_LayerNorm
from src.models.vit_dyt import ViT_DyT

logger = setup_logger("evaluate_log", save_to_file=True)

parser = argparse.ArgumentParser(description="Test and Evaluate ViT")
parser.add_argument("--model", type=str, choices=["layernorm", "dyt"], required=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--weights", type=str, required=True, help="Path for .pth file")
parser.add_argument(
    "--timestamp", type=str, required=True, help="Unique experiment identifier."
)
args = parser.parse_args()

log_name = f"{args.model}_evaluate_log"
logger = setup_logger(log_name, save_to_file=True)

CIFAR10_CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def load_test_data(data_dir: Path, batch_size: int) -> DataLoader:
    logger.info("Loading test_data.pt...")
    test_dataset = torch.load(data_dir / "test_data.pt", weights_only=False)

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def plot_confusion_matrix(conf_matrix, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(fpr, tpr, save_path):
    plt.figure(figsize=(10, 8))

    for i in range(10):
        plt.plot(fpr[i].cpu(), tpr[i].cpu(), lw=2, label=f"{CIFAR10_CLASSES[i]}")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves (One-vs-Rest) Multiclass")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting evaluation of the {args.model.upper()} model on {device}")

    # 1. Configuring Output Directories
    experiments_evaluate_dir = (
        Path("experiments") / "results" / f"{args.model}_{args.timestamp}"
    )
    experiments_evaluate_dir.mkdir(parents=True, exist_ok=True)

    # 2. Loading Data and Model
    data_dir = ROOT_DIR / "data" / "processed"
    test_loader = load_test_data(data_dir, args.batch_size)

    if args.model == "layernorm":
        model = ViT_LayerNorm().to(device)
    else:
        model = ViT_DyT().to(device)

    # 3. Loading Trained Weights
    logger.info(f"Loading weights from: {args.weights}")
    # ignores "trash" that thop library puts in the saved model
    # and only uses the important weights for the model
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()

    # 4. Initializing Metrics
    metric_acc = MulticlassAccuracy(num_classes=10).to(device)
    metric_f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)
    metric_conf_mat = MulticlassConfusionMatrix(num_classes=10).to(device)
    metric_roc = MulticlassROC(num_classes=10).to(device)

    logger.info("Performing inference on the test set...")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            metric_acc.update(preds, targets)
            metric_f1.update(preds, targets)
            metric_conf_mat.update(preds, targets)
            # ROC curve needs the raw logits (probabilities), not the final class
            metric_roc.update(logits, targets)

    # 5. Computing the Results
    final_acc = metric_acc.compute().item()
    final_f1 = metric_f1.compute().item()
    conf_matrix = metric_conf_mat.compute().cpu().numpy()
    fpr, tpr, thresholds = metric_roc.compute()

    logger.info(f"FINAL TEST RESULTS - {args.model.upper()}")
    logger.info(f"Accuracy: {final_acc:.4f}")
    logger.info(f"F1-Score: {final_f1:.4f}")

    # 6. Generating and Saving the Graphs
    cm_path = experiments_evaluate_dir / "confusion_matrix.png"
    plot_confusion_matrix(conf_matrix, cm_path)
    logger.info(f"Confusion Matrix saved to: {cm_path}")

    roc_path = experiments_evaluate_dir / "roc_curves.png"
    plot_roc_curves(fpr, tpr, roc_path)
    logger.info(f"ROC Curves saved to: {roc_path}")

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    evaluate()
