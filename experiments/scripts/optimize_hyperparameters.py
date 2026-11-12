"""
experiments/scripts/optimize_hyperparameters.py

Tier A.5: Hyperparameter Optimization Experiment
Optimizes lengthscale, variance, energy noise, and force noise using 
Negative Log Marginal Likelihood (NLML) across 3 different initializations.
Compares the optimized result against the fixed baseline.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import (
    energy_energy, energy_force, force_energy, force_force
)
from experiments.scripts.tune_soap_ethanol import get_normalized_soap

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

def nlml(phi, D_train, J_train, y_train):
    """Negative Log Marginal Likelihood with log-parameterization theta = exp(phi)"""
    l, v, nE, nF = np.exp(phi)
    
    # Soft bounds to prevent Cholesky collapse during L-BFGS-B line search
    if l < 0.1 or l > 10.0 or v < 0.01 or v > 100.0 or nE < 1e-5 or nF < 1e-5:
        return 1e10
        
    K = build_K(D_train, J_train, l, v, nE, nF)
    try:
        c, low = cho_factor(K, lower=True)
        alpha = cho_solve((c, low), y_train)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        return 0.5 * np.dot(y_train, alpha) + 0.5 * logdet
    except np.linalg.LinAlgError:
        return 1e10

def evaluate_rmse(D_train, J_train, y_train, D_test, J_test, y_test, l, v, nE, nF, E_std, dim):
    K_tr = build_K(D_train, J_train, l, v, nE, nF)
    c_tr, low = cho_factor(K_tr)
    K_star = build_K_star(D_test, J_test, D_train, J_train, l, v)
    alpha = cho_solve((c_tr, low), y_train)
    y_pred = K_star @ alpha
    
    err = (y_test - y_pred) * E_std
    idx_E = np.arange(0, len(y_test), dim)
    idx_F = np.setdiff1d(np.arange(len(y_test)), idx_E)
    
    e_rmse = np.sqrt(np.mean(err[idx_E]**2))
    f_rmse = np.sqrt(np.mean(err[idx_F]**2))
    return e_rmse, f_rmse

def main():
    print("--- Tier A.5: Marginal-Likelihood Hyperparameter Optimization ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3
    
    # Load Random Split configuration
    with open("experiments/configs/split_random.json", "r") as f:
        split = json.load(f)
        
    train_idx = split["pool"][:20] # 20 frames for optimization speed
    test_idx = split["test"][:50]  # 50 frames for evaluation
    
    print("Loading normalized SOAP features...")
    D_train, J_train = get_normalized_soap(z, R[train_idx], symbols, 5.0, 4, 3)
    D_test, J_test = get_normalized_soap(z, R[test_idx], symbols, 5.0, 4, 3)
    
    y_train = np.zeros(len(train_idx) * dim)
    for i, idx in enumerate(train_idx):
        y_train[i*dim] = E_norm[idx]
        y_train[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
        
    y_test = np.zeros(len(test_idx) * dim)
    for i, idx in enumerate(test_idx):
        y_test[i*dim] = E_norm[idx]
        y_test[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()

    # Define 3 distinct initializations in log-space: log([l, v, nE, nF])
    inits = [
        np.log([1.5, 1.0, 0.01, 0.05]),  # Init 1: Our previous "smoke-test" baseline
        np.log([0.5, 5.0, 0.05, 0.10]),  # Init 2: Short lengthscale, high variance
        np.log([3.0, 0.5, 0.005, 0.01])  # Init 3: Long lengthscale, low noise
    ]
    
    best_nlml = np.inf
    best_params = None
    
    print("\nStarting L-BFGS-B Optimization (3 Initializations)...")
    print(f"{'Init':<6}{'Final NLML':<15}{'l':<8}{'v':<8}{'nE':<8}{'nF':<8}{'Time(s)':<8}")
    print("-" * 60)
    
    for i, init_phi in enumerate(inits):
        t0 = time.time()
        res = minimize(nlml, init_phi, args=(D_train, J_train, y_train), 
                       method='L-BFGS-B', options={'maxiter': 50, 'disp': False})
        t_elap = time.time() - t0
        
        opt_l, opt_v, opt_nE, opt_nF = np.exp(res.x)
        print(f"{i+1:<6}{res.fun:<15.2f}{opt_l:<8.3f}{opt_v:<8.3f}{opt_nE:<8.4f}{opt_nF:<8.4f}{t_elap:<8.1f}")
        
        if res.fun < best_nlml:
            best_nlml = res.fun
            best_params = (opt_l, opt_v, opt_nE, opt_nF)
            
    print("\n--- Validation Comparison (Test Size: 50) ---")
    
    # Evaluate Fixed Baseline
    fix_l, fix_v, fix_nE, fix_nF = np.exp(inits[0])
    e_rmse_fix, f_rmse_fix = evaluate_rmse(D_train, J_train, y_train, D_test, J_test, y_test, 
                                           fix_l, fix_v, fix_nE, fix_nF, E_std, dim)
    print(f"Fixed Baseline -> Energy RMSE: {e_rmse_fix:.3f} kcal/mol | Force RMSE: {f_rmse_fix:.3f} kcal/mol/Å")
    
    # Evaluate Best Optimized
    opt_l, opt_v, opt_nE, opt_nF = best_params
    e_rmse_opt, f_rmse_opt = evaluate_rmse(D_train, J_train, y_train, D_test, J_test, y_test, 
                                           opt_l, opt_v, opt_nE, opt_nF, E_std, dim)
    print(f"Optimized      -> Energy RMSE: {e_rmse_opt:.3f} kcal/mol | Force RMSE: {f_rmse_opt:.3f} kcal/mol/Å")
    
    # Save optimized parameters
    opt_dict = {"l": opt_l, "v": opt_v, "nE": opt_nE, "nF": opt_nF}
    with open("experiments/configs/optimized_hyperparams.json", "w") as f:
        json.dump(opt_dict, f, indent=4)
    print("\nSaved best hyperparameters to experiments/configs/optimized_hyperparams.json")

if __name__ == "__main__":
    main()