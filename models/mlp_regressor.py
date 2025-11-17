"""
models/mlp_regressor.py

Fully-connected neural network (MLP) regressor for J1–J2 energies.

Input:  feature vector built from spin angles (theta, phi) or xyz.
Output: scalar energy prediction.
"""

import torch
import torch.nn as nn


class J1J2MLPRegressor(nn.Module):
    """
    Simple fully-connected regressor.

    input_dim:
        = 2 * n_spins  (angles mode)
        or 3 * n_spins (xyz mode)

    output_dim:
        = 1 for scalar energy prediction
    """

    def __init__(self, input_dim: int, hidden_dims=(64, 32, 16), output_dim: int = 1):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.net(x)
