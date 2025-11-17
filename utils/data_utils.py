"""
utils/data_utils.py

Data utilities such as train/validation/test splitting.
"""

from typing import Tuple

import torch


def train_val_test_split(
    x: torch.Tensor,
    y: torch.Tensor,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    shuffle: bool = True,
    seed: int = 0
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Split data into train / validation / test sets.

    Args:
        x: (N, ...) input tensor
        y: (N,) labels
        train_frac, val_frac, test_frac: fractions that should sum to 1.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1."

    N = x.shape[0]
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        x = x[perm]
        y = y[perm]

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    n_test = N - n_train - n_val

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
