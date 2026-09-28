import logging
import os
from datetime import datetime


def setup_logger(script_name: str, save_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(script_name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if save_to_file:
        os.makedirs("logs", exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(f"logs/{script_name}_{current_time}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
