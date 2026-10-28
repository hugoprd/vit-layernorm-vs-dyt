import sys
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codecarbon import EmissionsTracker
from thop import profile
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from logs.cmd.logger_config import setup_logger
from src.models.vit_layernorm import ViT_LayerNorm
from src.models.vit_dyt import ViT_DyT
from src.utils.experiment_extensions import EarlyStopping, log_system_specs

# 1. Hyperparameters and Config
parser = argparse.ArgumentParser(description="Training ViT CIFAR-10")
parser.add_argument("--model", type=str, choices=["layernorm", "dyt"], required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument(
    "--patience", type=int, default=16, help="Epochs of tolerance for Early Stopping"
)
parser.add_argument(
    "--timestamp", type=str, required=True, help="Unique experiment identifier."
)
args = parser.parse_args()

log_name = f"{args.model}_train_log"
logger = setup_logger(log_name, save_to_file=True)


def log_epoch_to_csv(filepath: Path, epoch: int, metrics: dict):
    """
    Saves the metrics of an epoch to a CSV file.
    If the file does not exist, it creates the headers automatically.
    """
    file_exists = filepath.is_file()

    row_data = {"epoch": epoch, **metrics}
    headers = list(row_data.keys())

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def log_summary_to_csv(filepath: Path, model_name: str, summary: dict):
    """
    Saves the final metrics (energy consumption, MACs, time) to the summary CSV file.
    """
    file_exists = filepath.is_file()

    row_data = {"model": model_name, **summary}
    headers = list(row_data.keys())

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def load_datasets(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Load processed PyTorch Dataset objects (Subsets) and create DataLoaders.
    """
    logger.info("Loading datasets (.pt)...")

    # weights_only=False permits load the Subset class that is saved at the .pt
    train_dataset = torch.load(data_dir / "train_data.pt", weights_only=False)
    val_dataset = torch.load(data_dir / "validation_data.pt", weights_only=False)

    # since they are already Dataset objects,
    # it's possible to pass them directly to the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Instances the chosen model.
    """
    if model_name == "layernorm":
        model = ViT_LayerNorm().to(device)
    elif model_name == "dyt":
        model = ViT_DyT().to(device)
    else:
        raise ValueError("Invalid model name. Choose 'layernorm' or 'dyt'.")

    return model


def calculate_macs_params(model, device, data_loader) -> tuple[float, float]:
    """
    Calculates the theoretical computational complexity of the model.
    """
    # extracts the first real batch of images and labels
    real_inputs, _ = next(iter(data_loader))

    # gets only the first image from the batch and adds the batch dimension
    # the shape changes from [3, 32, 32] to [1, 3, 32, 32]
    real_single_image = real_inputs[0].unsqueeze(0).to(device)

    # calculates the complexity using the real image from CIFAR-10
    macs, params = profile(model, inputs=(real_single_image,), verbose=False)
    logger.info(f"Complexity -> MACs: {macs/1e6:.2f} M | Parameters: {params/1e6:.2f} M")

    return macs, params


def train():
    experiments_training_dir = (
        Path("experiments") / "training" / f"{args.model}_{args.timestamp}"
    )
    experiments_training_dir.mkdir(parents=True, exist_ok=True)

    csv_metrics_path = experiments_training_dir / f"{args.model}_metrics_{args.timestamp}.csv"
    csv_summary_path = experiments_training_dir / f"training_cost_summary_{args.timestamp}.csv"

    log_system_specs(logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing experiment with model: {args.model.upper()} on {device}")

    # 2. Initializing Energy Tracker (CodeCarbon)
    tracker = EmissionsTracker(
        project_name=f"vit_{args.model}",
        log_level="error",
        output_dir=str(experiments_training_dir),
    )
    tracker.start()

    # 3. Data and Model
    data_dir = ROOT_DIR / "data" / "processed"
    train_loader, val_loader = load_datasets(data_dir, args.batch_size)

    model = get_model(args.model, device)

    # theoric complexity
    macs, params = calculate_macs_params(model, device, train_loader)

    # 4. Optimizer, Loss and Metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    metric_acc = MulticlassAccuracy(num_classes=10).to(device)
    metric_f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)

    # 5. Training Loop
    model_save_path = experiments_training_dir / f"{args.model}_best_weights.pth"

    early_stopping = EarlyStopping(patience=args.patience)

    experiment_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        # loop over batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        metric_acc.reset()
        metric_f1.reset()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                val_loss += criterion(outputs, targets).item()

                preds = torch.argmax(outputs, dim=1)
                metric_acc.update(preds, targets)
                metric_f1.update(preds, targets)

        # epoch final metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = metric_acc.compute().item()
        val_f1 = metric_f1.compute().item()

        logger.info(
            f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Train Loss: {avg_train_loss:.4f} | "
            f"Throughput: {throughput:.1f} img/s"
        )

        epoch_metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "throughput_imgs_per_sec": throughput,
        }

        log_epoch_to_csv(csv_metrics_path, epoch, epoch_metrics)

        is_best = early_stopping(avg_val_loss)

        if is_best:
            torch.save(model.state_dict(), model_save_path)

        if early_stopping.early_stop:
            logger.info(
                f"Early stopping activated. No improvements for {args.patience} epochs."
            )
            logger.info(f"Training interrupted at epoch {epoch}.")

            break

    # 7. Stopping Energy Tracker
    emissions = tracker.stop()

    total_time_seconds = time.time() - experiment_start_time

    logger.info(f"Total energy cost: {tracker._total_energy.kWh:.6f} kWh")
    logger.info(f"Carbon footprint: {emissions:.6f} kg CO2eq")
    logger.info(f"Total training time: {total_time_seconds:.2f} seconds")

    summary_metrics = {
        "macs_millions": macs / 1e6,
        "parameters_millions": params / 1e6,
        "total_energy_kwh": tracker._total_energy.kWh,
        "carbon_emissions_kgCO2": emissions,
        "total_time_seconds": total_time_seconds,
    }

    log_summary_to_csv(csv_summary_path, args.model, summary_metrics)

    logger.info(f"Model weights saved to: {model_save_path}")
    logger.info(f"Metrics saved locally to: {csv_metrics_path}")
    logger.info(f"Summary cost metrics saved locally to: {csv_summary_path}")
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    train()
