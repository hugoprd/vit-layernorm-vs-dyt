import platform
import psutil

import torch


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
