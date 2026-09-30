import sys
import copy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from logs.cmd.logger_config import setup_logger

LOG_NAME = "process_data_log"
logger = setup_logger(LOG_NAME, save_to_file=False)


def _get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the CIFAR-10 dataset from data/raw/, splits the validation set,
    saves the processed datasets to data/processed/, and returns the ready DataLoaders.
    """
    raw_dir = ROOT_DIR / "data" / "raw"
    processed_dir = ROOT_DIR / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from directory: {raw_dir}")

    # 1. Definition of transformations for training and evaluation
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    try:
        # 2. Load the raw training dataset (all 5 batches)
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=str(raw_dir), train=True, download=False, transform=transform_train
        )

        # cloning the dataset specifically for validation so its possible
        # to apply transform_eval safely
        val_dataset_base = copy.deepcopy(full_train_dataset)
        val_dataset_base.transform = transform_eval

        # 3. Load the official test dataset
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(raw_dir), train=False, download=False, transform=transform_eval
        )

        # 4. Create the train and validation split (45k | 5k) using indices
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()

        train_idx = indices[:45000]
        val_idx = indices[45000:]

        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(val_dataset_base, val_idx)

        logger.info(
            f"Split completed: Train={len(train_dataset)}, "
            f"Validation={len(val_dataset)}, Test={len(test_dataset)}"
        )

        # 5. Saves the processed datasets
        logger.info(f"Saving processed datasets to: {processed_dir}")
        torch.save(train_dataset, processed_dir / "train_data.pt")
        torch.save(val_dataset, processed_dir / "validation_data.pt")
        torch.save(test_dataset, processed_dir / "test_data.pt")
        logger.info("Saved datasets successfully.")

        # 6. DataLoaders creation just to visualization
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.exception(f"Error to process and load the datasets: {e}")
        raise e


def process_data():
    logger.info("Starting test of the data pipeline...")
    train_loader, val_loader, test_loader = _get_dataloaders()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    logger.info(
        "Train batch loaded successfully. "
        f"Image shape: {train_images.shape}, Label shape: {train_labels.shape}"
    )

    logger.info(
        "Validation batch loaded successfully. "
        f"Image shape: {val_images.shape}, Label shape: {val_labels.shape}"
    )

    logger.info(
        "Test batch loaded successfully. "
        f"Image shape: {test_images.shape}, Label shape: {test_labels.shape}"
    )


if __name__ == "__main__":
    process_data()
