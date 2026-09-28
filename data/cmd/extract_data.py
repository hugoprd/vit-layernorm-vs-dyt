import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import os
import tarfile
import urllib.request
from logs.cmd.logger_config import setup_logger

LOG_NAME = "extract_data_log"
logger = setup_logger(LOG_NAME, save_to_file=False)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_last_logged_percent = -1


def _report_progress(block_num, block_size, total_size):
    global _last_logged_percent
    downloaded = block_num * block_size

    if total_size > 0:
        percent = min(int(downloaded * 100 / total_size), 100)

        if percent != _last_logged_percent and percent % 10 == 0:
            _last_logged_percent = percent
            logger.info(
                "Downloading dataset... "
                f"{percent}% completed ({downloaded / (1024*1024):.1f} MB)"
            )


def extract_data():
    """
    Extracts the CIFAR-10 dataset and saves it to the data/raw/ directory.
    """
    target_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(target_dir, exist_ok=True)

    tar_path = os.path.join(target_dir, "cifar-10-python.tar.gz")
    extracted_folder = os.path.join(target_dir, "cifar-10-batches-py")

    if os.path.exists(extracted_folder):
        logger.info("CIFAR-10 Dataset is already downloaded and extracted in data/raw/!")
        return

    try:
        logger.info(f"Starting download of: {CIFAR_URL}")

        urllib.request.urlretrieve(CIFAR_URL, tar_path, reporthook=_report_progress)
        logger.info("Download completed successfully. Starting extraction...")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)

        logger.info(f"Extraction completed successfully in: {target_dir}")

        if os.path.exists(tar_path):
            os.remove(tar_path)
            logger.info("Temporary compressed file removed.")
    except Exception as e:
        logger.exception(f"Error while downloading or extracting the dataset: {e}")


if __name__ == "__main__":
    extract_data()
