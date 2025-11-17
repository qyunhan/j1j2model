"""
physics/j1j2_solver.py

Exact solver for the 1D quantum J1–J2 Heisenberg model.

This is the "physics engine" of the project:
- Builds the Hamiltonian H for a chain of n_spins.
- Converts spin angles (theta, phi) into product states.
- Computes exact energies E = ⟨ψ|H|ψ⟩.
- Generates supervised training data (theta, phi, energy).
"""

from typing import Tuple

import numpy as np
import torch


class QuantumJ1J2Solver:
    """
    Exact quantum solver for 1D J1–J2 Heisenberg model with n_spins sites.

    Hamiltonian:
        H = J1 * Σ_{<i,j>} S_i · S_j  +  J2 * Σ_{<<i,j>>} S_i · S_j
    where
        S_i = (σ_i^x, σ_i^y, σ_i^z) / 2
    and σ_i^α are Pauli matrices acting on site i.
    """

    def __init__(self, n_spins: int = 3, J1: float = 1.0, J2: float = 0.5):
        self.n_spins = n_spins
        self.J1 = J1
        self.J2 = J2
        self.dim = 2 ** n_spins  # Hilbert-space dimension (each spin has 2 states)

        # Pauli matrices (2x2, complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)

        # Build and diagonalize the Hamiltonian once
        self.H = self._build_hamiltonian()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        self.ground_state_energy = float(self.eigenvalues[0].real)
        self.ground_state = self.eigenvectors[:, 0]

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _tensor_product(self, matrices):
        """
        Compute Kronecker product of a list of matrices.

        For n spins, operators on the full system are (2^n x 2^n).
        """
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result

    def _spin_operator(self, site: int, pauli: str):
        """
        Return S_site^pauli on the full Hilbert space.

        Args:
            site:  which spin index (0, 1, ..., n_spins-1)
            pauli: 'x', 'y', or 'z'
        """
        pauli_dict = {'x': self.sigma_x, 'y': self.sigma_y, 'z': self.sigma_z}

        mats = []
        for i in range(self.n_spins):
            if i == site:
                mats.append(pauli_dict[pauli])  # Pauli on this site
            else:
                mats.append(self.identity)      # Identity otherwise

        # S = (1/2)σ
        return 0.5 * self._tensor_product(mats)

    def _build_hamiltonian(self):
        """
        Construct the full J1–J2 Heisenberg Hamiltonian matrix H.
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # J1: nearest neighbours (open chain)
        for i in range(self.n_spins - 1):
            j = i + 1
            for pauli in ['x', 'y', 'z']:
                H += self.J1 * (self._spin_operator(i, pauli) @
                                self._spin_operator(j, pauli))

        # J2: next-nearest neighbours
        if self.n_spins >= 3:
            for i in range(self.n_spins - 2):
                j = i + 2
                for pauli in ['x', 'y', 'z']:
                    H += self.J2 * (self._spin_operator(i, pauli) @
                                    self._spin_operator(j, pauli))

        return H

    def _angles_to_state(self, theta: np.ndarray, phi: np.ndarray):
        """
        Build product state:
            |ψ⟩ = ⊗_i ( cos(θ_i/2) |↑⟩ + e^{iφ_i} sin(θ_i/2) |↓⟩ )

        theta, phi: shape (n_spins,)
        """
        # Allow torch tensors as input
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
            phi = phi.detach().cpu().numpy()

        def one_spin(th, ph):
            return np.array([
                np.cos(th / 2.0),
                np.exp(1j * ph) * np.sin(th / 2.0)
            ], dtype=complex)

        psi = one_spin(theta[0], phi[0])
        for i in range(1, self.n_spins):
            psi = np.kron(psi, one_spin(theta[i], phi[i]))
        return psi

    # ------------------------------
    # Public methods
    # ------------------------------

    def compute_energy_for_angles(self, theta: np.ndarray, phi: np.ndarray) -> float:
        """
        Compute energy E = ⟨ψ| H |ψ⟩ for the product state defined by (theta, phi).
        """
        psi = self._angles_to_state(theta, phi)
        E = np.vdot(psi, self.H @ psi)  # conjugate(psi) dot (H psi)
        return float(E.real)

    def get_training_data(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample product states and compute their exact energies.

        Returns:
            theta:    (n_samples, n_spins) in [0, π]
            phi:      (n_samples, n_spins) in [0, 2π)
            energies: (n_samples,)
        """
        theta = np.random.rand(n_samples, self.n_spins) * np.pi
        phi = np.random.rand(n_samples, self.n_spins) * 2 * np.pi

        energies = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            energies[i] = self.compute_energy_for_angles(theta[i], phi[i])

        return (
            torch.tensor(theta, dtype=torch.float32),
            torch.tensor(phi, dtype=torch.float32),
            torch.tensor(energies, dtype=torch.float32)
        )
