"""
experiments/scripts/fit_ethanol.py

Step 1: Fit a derivative-GP to rMD17 ethanol.
Precomputes SOAP features, caches them, logs optimization progress, 
and produces the learning curve safely.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from ase import Atoms
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import (
    soap_with_jacobian, energy_energy, energy_force, force_energy, force_force
)

CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cached_soap(z, R_subset, symbols):
    """Computes or loads cached SOAP descriptors and Jacobians for a subset of frames."""
    cache_path = CACHE_DIR / f"soap_cache_N{len(R_subset)}.npz"
    if cache_path.exists():
        print(f"Loading cached SOAP features from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return [np.array(d, dtype=float) for d in data["D"]], [np.array(j, dtype=float) for j in data["J"]]
    
    print(f"Computing SOAP features for {len(R_subset)} frames (this will take ~10 seconds)...")
    soap = SOAP(species=symbols, r_cut=5.0, n_max=4, l_max=3, sparse=False)
    D_list, J_list = [], []
    for idx, coords in enumerate(R_subset):
        d, j = soap_with_jacobian(soap, Atoms(numbers=z, positions=coords))
        D_list.append(d)
        J_list.append(j)
        if (idx + 1) % 10 == 0 or idx == len(R_subset) - 1:
            print(f"  Processed {idx + 1}/{len(R_subset)} frames")
            
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
    print("Loading rMD17 Ethanol...")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    
    E_mean, E_std = np.mean(E), np.std(E)
    E_norm = (E - E_mean) / E_std
    F_norm = F / E_std
    
    symbols = ["C", "O", "H"]
    n_atoms = len(z)
    dim = 1 + n_atoms * 3
    
    test_idx = np.arange(50000, 50020)
    D_test, J_test = get_cached_soap(z, R[test_idx], symbols)
    
    y_test = np.zeros(len(test_idx) * dim)
    for i, idx in enumerate(test_idx):
        y_test[i*dim] = E_norm[idx]
        y_test[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
        
    N_train_list = [5, 10, 20]
    rmse_E, rmse_F = [], []
    
    for N in N_train_list:
        print(f"\n--- Training on N={N} frames ---")
        train_idx = np.arange(0, N)
        D_train, J_train = get_cached_soap(z, R[train_idx], symbols)
        
        y_train = np.zeros(N * dim)
        for i, idx in enumerate(train_idx):
            y_train[i*dim] = E_norm[idx]
            y_train[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
            
        def nll(log_params):
            l, v, nE, nF = np.exp(log_params)
            K = build_K(D_train, J_train, l, v, nE, nF)
            try:
                c, low = cho_factor(K, overwrite_a=False, check_finite=False)
                alpha = cho_solve((c, low), y_train, check_finite=False)
                logdet = 2.0 * np.sum(np.log(np.diag(c)))
                val = 0.5 * np.dot(y_train, alpha) + 0.5 * logdet + 0.5 * len(y_train) * np.log(2 * np.pi)
                print(f"  [NLL] l={l:.2f}, v={v:.2f} -> Loss: {val:.2f}", end='\r')
                return val
            except np.linalg.LinAlgError:
                return 1e15

        print("Optimizing hyperparameters (L-BFGS-B)...")
        t0 = time.time()
        init_log_params = np.log([2.5, 1.0, 0.05, 0.05])
        
        res = minimize(nll, init_log_params, method='L-BFGS-B', 
                       bounds=[(np.log(0.5), np.log(10.0)), (np.log(0.1), np.log(10.0)), 
                               (np.log(1e-3), np.log(0.5)), (np.log(1e-3), np.log(0.5))],
                       options={'maxiter': 30, 'disp': False})
        
        opt_l, opt_v, opt_nE, opt_nF = np.exp(res.x)
        print(f"\nDone in {time.time()-t0:.1f}s. Opt: l={opt_l:.3f}, v={opt_v:.3f}")
        
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
        print(f"Test RMSE -> Energy: {e_rmse:.3f} kcal/mol | Force: {f_rmse:.3f} kcal/mol/A")
        
        rmse_E.append(e_rmse)
        rmse_F.append(f_rmse)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(N_train_list, rmse_E, 'o-', color='C0')
    plt.xlabel('Number of Training Frames (N)')
    plt.ylabel('Energy RMSE (kcal/mol)')
    plt.title('Energy Learning Curve')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(N_train_list, rmse_F, 'o-', color='C1')
    plt.xlabel('Number of Training Frames (N)')
    plt.ylabel('Force RMSE (kcal/mol/Å)')
    plt.title('Force Learning Curve')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('figures/ethanol_learning_curve.png', dpi=300)
    print("\nSaved learning curve to figures/ethanol_learning_curve.png")

if __name__ == "__main__":
    main()