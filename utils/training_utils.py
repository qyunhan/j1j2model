"""
utils/training_utils.py

Shared training utilities for both MLP and CNN:

- train_one_epoch: standard supervised training loop for one epoch
- evaluate: compute MSE and MAE on a dataset
- build_optimizer: choose optimizer (Adam, SGD, AdamW) with weight decay
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> float:
    """
    Train the model for one epoch on the training set.
    Returns the average training loss (MSE).
    """
    model.train()
    N = x_train.shape[0]
    epoch_loss = 0.0

    # Shuffle indices
    perm = torch.randperm(N)
    x_train = x_train[perm]
    y_train = y_train[perm]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x_train[start:end].to(device)
        yb = y_train[start:end].to(device)

        optimizer.zero_grad()
        preds = model(xb).squeeze(-1)  # (batch_size,)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * (end - start)

    return epoch_loss / N


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device
) -> Tuple[float, float]:
    """
    Compute loss (MSE) and MAE on a given dataset.
    """
    model.eval()
    N = x.shape[0]
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for start in range(0, N, 256):
            end = min(start + 256, N)
            xb = x[start:end].to(device)
            yb = y[start:end].to(device)

            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            mae = torch.mean(torch.abs(preds - yb))

            total_loss += loss.item() * (end - start)
            total_mae += mae.item() * (end - start)

    avg_loss = total_loss / N
    avg_mae = total_mae / N
    return avg_loss, avg_mae


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float
) -> optim.Optimizer:
    """
    Small factory to create optimizer based on a string.

    Lets you experiment with different optimization schemes.
    """
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer name. Choose from: 'adam', 'sgd', 'adamw'.")
