import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_cuda_info():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    else:
        print("CUDA is available.")
        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("GPU count:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            print(f"\nGPU {i + 1} Properties:")
            print("Name:", torch.cuda.get_device_name(device))
            print("Capability:", torch.cuda.get_device_capability(device))
            print(
                "Total Memory:",
                round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
                "GB",
            )
            print(
                "CUDA Compute Capability:",
                torch.cuda.get_device_properties(device).major,
                torch.cuda.get_device_properties(device).minor,
            )
            print("-----------------------------------------------------------------")


def today_time_str() -> str:
    return datetime.today().strftime("%Y_%m_%d_%H_%M")


def set_logger(filename: str):
    logging.basicConfig(filename=filename, level=logging.INFO, format="%(asctime)s %(message)s")
