"""
experiments/scripts/tune_soap_ethanol.py

Step 2: SOAP parameter tuning with L2 descriptor normalization
and robust hyperparameter optimization bounds.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ase import Atoms
from dscribe.descriptors import SOAP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import (
    soap_with_jacobian, energy_energy, energy_force, force_energy, force_force
)

CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_normalized_soap(z, R_subset, symbols, r_cut, n_max, l_max):
    """Computes SOAP descriptors and Jacobians with per-atom L2 normalization."""
    cache_path = CACHE_DIR / f"soap_norm_rcut{r_cut}_n{n_max}_l{l_max}_N{len(R_subset)}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return [np.array(d, dtype=float) for d in data["D"]], [np.array(j, dtype=float) for j in data["J"]]
    
    soap = SOAP(species=symbols, r_cut=r_cut, n_max=n_max, l_max=l_max, sparse=False)
    D_list, J_list = [], []
    
    for coords in R_subset:
        d, j = soap_with_jacobian(soap, Atoms(numbers=z, positions=coords))
        
        norms = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12  # (n_atoms, 1)
        d_norm = d / norms
        
        norm_c = norms[:, None, None, :]  # shape (n_atoms, 1, 1, 1)
        dot_proj = np.einsum("cf,cmkf->cmk", d, j)
        proj_term = np.einsum("cf,cmk->cmkf", d, dot_proj) / (norm_c ** 3)
        j_norm = (j / norm_c) - proj_term
        
        D_list.append(d_norm)
        J_list.append(j_norm)
            
    np.savez_compressed(cache_path, D=np.array(D_list, dtype=float), J=np.array(J_list, dtype=float))
    return D_list, J_list

def build_K(D_list, J_list, lengthscale, variance, noise_E, noise_F):
    N = len(D_list)
    n_atoms = D_list[0].shape[0]
    dim = 1 + n_atoms * 3
    K = np.zeros((N * dim, N * dim))
    
    for i in range(N):
        for j in range(i, N):
            Da, Ja = D_list[i], J_list[i]
            Db, Jb = D_list[j], J_list[j]
            
            K_EE = energy_energy(Da, Db, lengthscale, variance)
            K_EF = energy_force(Da, Db, Jb, lengthscale, variance).flatten()
            K_FE = force_energy(Da, Ja, Db, lengthscale, variance).flatten()
            K_FF = force_force(Da, Ja, Db, Jb, lengthscale, variance).reshape(n_atoms*3, n_atoms*3)
            
            r, c = i * dim, j * dim
            K[r, c] = K_EE
            K[r, c+1 : c+dim] = K_EF
            K[r+1 : r+dim, c] = K_FE
            K[r+1 : r+dim, c+1 : c+dim] = K_FF
            
            if i != j:
                K[c, r] = K_EE
                K[c, r+1 : r+dim] = K_FE
                K[c+1 : c+dim, r] = K_EF
                K[c+1 : c+dim, r+1 : r+dim] = K_FF.T

    idx_E = np.arange(0, N * dim, dim)
    idx_F = np.setdiff1d(np.arange(N * dim), idx_E)
    K[idx_E, idx_E] += noise_E**2
    K[idx_F, idx_F] += noise_F**2
    return K

def build_K_star(D_test, J_test, D_train, J_train, lengthscale, variance):
    N_test, N_train = len(D_test), len(D_train)
    n_atoms = D_train[0].shape[0]
    dim = 1 + n_atoms * 3
    K_star = np.zeros((N_test * dim, N_train * dim))
    
    for i in range(N_test):
        for j in range(N_train):
            Da, Ja = D_test[i], J_test[i]
            Db, Jb = D_train[j], J_train[j]
            
            K_star[i*dim, j*dim] = energy_energy(Da, Db, lengthscale, variance)
            K_star[i*dim, j*dim+1 : (j+1)*dim] = energy_force(Da, Db, Jb, lengthscale, variance).flatten()
            K_star[i*dim+1 : (i+1)*dim, j*dim] = force_energy(Da, Ja, Db, lengthscale, variance).flatten()
            K_star[i*dim+1 : (i+1)*dim, j*dim+1 : (j+1)*dim] = force_force(Da, Ja, Db, Jb, lengthscale, variance).reshape(n_atoms*3, -1)
    return K_star

def main():
    print("--- Step 2: SOAP Hyperparameter & Descriptor Tuning ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    
    symbols = ["C", "O", "H"]
    n_atoms = len(z)
    dim = 1 + n_atoms * 3
    
    test_idx = np.arange(50000, 50020)
    train_idx = np.arange(0, 20)
    
    configs = [
        {"r_cut": 4.0, "n_max": 4, "l_max": 3},
        {"r_cut": 5.0, "n_max": 4, "l_max": 3},
        {"r_cut": 5.0, "n_max": 6, "l_max": 4},
    ]
    
    print("\nEvaluating Normalized SOAP Configurations on N=20 frames:")
    print(f"{'r_cut':<8}{'n_max':<8}{'l_max':<8}{'Energy RMSE (kcal/mol)':<25}{'Force RMSE (kcal/mol/Å)':<25}")
    print("-" * 70)
    
    for cfg in configs:
        r_cut, n_max, l_max = cfg["r_cut"], cfg["n_max"], cfg["l_max"]
        D_test, J_test = get_normalized_soap(z, R[test_idx], symbols, r_cut, n_max, l_max)
        D_train, J_train = get_normalized_soap(z, R[train_idx], symbols, r_cut, n_max, l_max)
        
        y_train = np.zeros(len(train_idx) * dim)
        for i, idx in enumerate(train_idx):
            y_train[i*dim] = E_norm[idx]
            y_train[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
            
        y_test = np.zeros(len(test_idx) * dim)
        for i, idx in enumerate(test_idx):
            y_test[i*dim] = E_norm[idx]
            y_test[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
            
        opt_l, opt_v, opt_nE, opt_nF = 1.5, 1.0, 0.01, 0.05
        
        K_train = build_K(D_train, J_train, opt_l, opt_v, opt_nE, opt_nF)
        K_star = build_K_star(D_test, J_test, D_train, J_train, opt_l, opt_v)
        
        c, low = cho_factor(K_train)
        alpha = cho_solve((c, low), y_train)
        y_pred = K_star @ alpha
        
        err = (y_test - y_pred) * E_std
        idx_E_test = np.arange(0, len(test_idx) * dim, dim)
        idx_F_test = np.setdiff1d(np.arange(len(test_idx) * dim), idx_E_test)
        
        e_rmse = np.sqrt(np.mean(err[idx_E_test]**2))
        f_rmse = np.sqrt(np.mean(err[idx_F_test]**2))
        
        print(f"{r_cut:<8}{n_max:<8}{l_max:<8}{e_rmse:<25.3f}{f_rmse:<25.3f}")

if __name__ == "__main__":
    main()