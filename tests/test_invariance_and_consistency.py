"""
tests/test_invariance_and_consistency.py

Comprehensive verification suite for the Energy-Force GP kernel.
Locks in Tier A.1 (Verification) and Tier D.17 (Symmetry) of the roadmap.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from dscribe.descriptors import SOAP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kernels.soap_ef_kernel import (
    soap_with_jacobian, energy_energy, energy_force, force_energy, force_force
)

@pytest.fixture
def setup_data():
    """Creates a basic water molecule and SOAP setup for testing."""
    soap = SOAP(species=["H", "O"], r_cut=5.0, n_max=3, l_max=2, sparse=False)
    
    # Standard H2O configuration
    coords = np.array([
        [0.0, 0.0, 0.0],          # O
        [0.757, 0.586, 0.0],      # H
        [-0.757, 0.586, 0.0]      # H
    ])
    symbols = ["O", "H", "H"]
    atoms = Atoms(symbols=symbols, positions=coords)
    
    D, J = soap_with_jacobian(soap, atoms)
    
    # L2 normalization matching our active learning pipeline
    norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
    D_norm = D / norms
    norm_c = norms[:, None, None, :]
    dot_proj = np.einsum("cf,cmkf->cmk", D, J)
    proj_term = np.einsum("cf,cmk->cmkf", D, dot_proj) / (norm_c ** 3)
    J_norm = (J / norm_c) - proj_term
    
    return atoms, soap, D_norm, J_norm, coords

def test_positive_semidefiniteness(setup_data):
    """Tier A.1: Ensures the full joint covariance matrix is PSD."""
    _, _, D, J, _ = setup_data
    n_atoms = D.shape[0]
    dim = 1 + n_atoms * 3
    l, v = 1.5, 1.0
    
    K = np.zeros((dim, dim))
    K[0, 0] = energy_energy(D, D, l, v)
    K[0, 1:] = energy_force(D, D, J, l, v).flatten()
    K[1:, 0] = force_energy(D, J, D, l, v).flatten()
    K[1:, 1:] = force_force(D, J, D, J, l, v).reshape(n_atoms*3, n_atoms*3)
    
    # A symmetric matrix is PSD if all eigenvalues are >= 0
    eigvals = np.linalg.eigvalsh(K)
    min_eig = np.min(eigvals)
    
    assert min_eig > -1e-10, f"Matrix is not PSD. Minimum eigenvalue: {min_eig}"
    # Verify symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-12, err_msg="Covariance matrix is not symmetric.")

def test_gradient_consistency(setup_data):
    """Tier A.1: Verifies F = -∇_R E via finite differences on the kernel."""
    atoms, soap, D, J, coords = setup_data
    l, v = 1.5, 1.0
    
    # Analytical Energy-Force covariance
    K_EF_analytical = energy_force(D, D, J, l, v) # Shape: (n_atoms, 3)
    
    # Numerical Energy-Force covariance (finite difference on Energy-Energy)
    eps = 1e-5
    K_EF_numerical = np.zeros_like(K_EF_analytical)
    
    for m in range(len(coords)):
        for k in range(3):
            coords_plus = coords.copy()
            coords_plus[m, k] += eps
            atoms_plus = Atoms(symbols=atoms.symbols, positions=coords_plus)
            # FIX: Use soap.create() to get just the descriptor cleanly
            D_plus = soap.create(atoms_plus)
            D_plus = D_plus / (np.linalg.norm(D_plus, axis=1, keepdims=True) + 1e-12)
            
            coords_minus = coords.copy()
            coords_minus[m, k] -= eps
            atoms_minus = Atoms(symbols=atoms.symbols, positions=coords_minus)
            # FIX: Use soap.create() to get just the descriptor cleanly
            D_minus = soap.create(atoms_minus)
            D_minus = D_minus / (np.linalg.norm(D_minus, axis=1, keepdims=True) + 1e-12)
            
            K_plus = energy_energy(D, D_plus, l, v)
            K_minus = energy_energy(D, D_minus, l, v)
            
            # F_b = -∇_R_b E, so Cov(E_a, F_b) = -∇_R_b Cov(E_a, E_b)
            K_EF_numerical[m, k] = -(K_plus - K_minus) / (2 * eps)
            
    np.testing.assert_allclose(K_EF_analytical, K_EF_numerical, atol=1e-5, 
                               err_msg="Analytical K_EF does not match finite difference of K_EE.")
def test_translation_invariance(setup_data):
    """Tier D.17: Verifies kernel output is invariant to rigid translations."""
    atoms, soap, D, J, coords = setup_data
    l, v = 1.5, 1.0
    
    # Apply rigid translation
    shift = np.array([5.0, -3.2, 1.1])
    atoms_shifted = Atoms(symbols=atoms.symbols, positions=coords + shift)
    
    D_shift, J_shift = soap_with_jacobian(soap, atoms_shifted)
    D_shift = D_shift / (np.linalg.norm(D_shift, axis=1, keepdims=True) + 1e-12)
    
    K_EE_orig = energy_energy(D, D, l, v)
    K_EE_shift = energy_energy(D_shift, D_shift, l, v)
    
    np.testing.assert_allclose(K_EE_orig, K_EE_shift, atol=1e-10, 
                               err_msg="Kernel is not invariant to translation.")

def test_rotation_invariance(setup_data):
    """Tier D.17: Verifies scalar energy kernel is invariant to rigid 3D rotations."""
    atoms, soap, D, J, coords = setup_data
    l, v = 1.5, 1.0
    
    # 90 degree rotation around Z axis
    theta = np.pi / 2
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,              1]
    ])
    
    coords_rotated = coords @ rot_z.T
    atoms_rotated = Atoms(symbols=atoms.symbols, positions=coords_rotated)
    
    D_rot, _ = soap_with_jacobian(soap, atoms_rotated)
    D_rot = D_rot / (np.linalg.norm(D_rot, axis=1, keepdims=True) + 1e-12)
    
    K_EE_orig = energy_energy(D, D, l, v)
    K_EE_rot = energy_energy(D_rot, D_rot, l, v)
    
    np.testing.assert_allclose(K_EE_orig, K_EE_rot, atol=1e-10, 
                               err_msg="Kernel is not invariant to rotation.")