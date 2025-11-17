"""
utils/feature_utils.py

Helper functions to convert (theta, phi) spin angles into:

- Flat features for MLP  (angles or xyz)
- 3-channel features for CNN (x, y, z) on a 1D spin chain
"""

from typing import Tuple

import torch


def angles_to_xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert angles to 3D spin vectors.

    Inputs:
        theta: (batch_size, n_spins)
        phi:   (batch_size, n_spins)

    For each spin:
        x = sin θ cos φ
        y = sin θ sin φ
        z = cos θ

    Output:
        xyz: (batch_size, 3 * n_spins)
             [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    xyz = torch.stack([x, y, z], dim=-1)   # (B, n_spins, 3)
    xyz = xyz.view(theta.shape[0], -1)     # (B, 3 * n_spins)
    return xyz


def make_inputs_from_angles(
    theta: torch.Tensor,
    phi: torch.Tensor,
    mode: str = "angles"
) -> Tuple[torch.Tensor, int]:
    """
    Build MLP input features from (theta, phi).

    mode='angles':
        input = [θ1..θN, φ1..φN] → size = 2 * n_spins

    mode='xyz':
        input = [x1,y1,z1, ..., xN,yN,zN] → size = 3 * n_spins

    Returns:
        x:         feature tensor (B, input_dim)
        input_dim: number of features per sample
    """
    if mode == "angles":
        x = torch.cat([theta, phi], dim=1)   # (B, 2 * n_spins)
    elif mode == "xyz":
        x = angles_to_xyz(theta, phi)        # (B, 3 * n_spins)
    else:
        raise ValueError("mode must be 'angles' or 'xyz'")

    return x, x.shape[1]


def angles_to_cnn_input(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert angles to 1D CNN input.

    θ, φ: (B, n_spins)

    Output:
        cnn_input: (B, 3, n_spins)
        where channels are [x, y, z].
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    cnn_input = torch.stack([x, y, z], dim=1)  # (B, 3, n_spins)
    return cnn_input
