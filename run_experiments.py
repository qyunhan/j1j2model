# run_experiments.py
"""
Entry-point script for the J1–J2 spin-chain project.

- Sets all random seeds for reproducibility.
- Runs the MLP and/or CNN training scripts.
- Saves training curves into the 'plots/' directory.

Usage (from the project root):
    python run_experiments.py
"""

import os
import random

import numpy as np
import torch

# Import the training entry points
from train.train_mlp import main as run_mlp
from train.train_cnn import main as run_cnn


def set_seed(seed: int = 123):
    """
    Set random seeds for Python, NumPy and PyTorch
    to make experiments as reproducible as possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Extra flags for more deterministic behaviour on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")


if __name__ == "__main__":
    # 1. Set global seed
    set_seed(123)

    # 2. Run MLP experiment
    print("\n[INFO] Running MLP experiment...\n")
    run_mlp()

    # 3. Run CNN experiment
    print("\n[INFO] Running CNN experiment...\n")
    run_cnn()

    print("\n[INFO] All experiments finished. Check the 'plots/' folder for results.")
