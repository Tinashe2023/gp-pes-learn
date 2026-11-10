"""
experiments/scripts/active_learning_loop.py

Step 4: Active Learning Benchmarking on Ethanol C-O Bond Gap
Compares Random, Max-Variance (Trace), and Log-Determinant Information Gain (D-Optimality).
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import time

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

def compute_predictive_covariance_block(D_cand, J_cand, D_train, J_train, c_train, l, v):
    """Computes full 28x28 predictive covariance matrix for candidate frame."""
    dim = 1 + 9 * 3
    K_star = build_K_star([D_cand], [J_cand], D_train, J_train, l, v)
    
    K_self = np.zeros((dim, dim))
    K_self[0, 0] = energy_energy(D_cand, D_cand, l, v)
    K_self[0, 1:] = energy_force(D_cand, D_cand, J_cand, l, v).flatten()
    K_self[1:, 0] = force_energy(D_cand, J_cand, D_cand, l, v).flatten()
    K_self[1:, 1:] = force_force(D_cand, J_cand, D_cand, J_cand, l, v).reshape(27, 27)
    
    v_solve = cho_solve((c_train, False), K_star.T)
    Sigma_cand = K_self - K_star @ v_solve
    return Sigma_cand

def compute_acquisition_score(strategy, D_cand, J_cand, D_train, J_train, c_train, l, v, nE, nF):
    Sigma_cand = compute_predictive_covariance_block(D_cand, J_cand, D_train, J_train, c_train, l, v)
    
    if strategy == "max_variance":
        return np.trace(Sigma_cand)
    elif strategy == "info_gain": # D-Optimality Log-Det
        Omega = np.diag([nE**2] + [nF**2]*27)
        sign, logdet = np.linalg.slogdet(Sigma_cand + Omega)
        return logdet if sign > 0 else -1e10

def run_active_learning(strategy, D_pool, J_pool, y_pool, D_test, J_test, y_test, E_std, n_init=5, n_steps=10):
    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05
    dim = 1 + 9 * 3
    
    rng = np.random.default_rng(42)
    selected_idx = list(rng.choice(len(D_pool), size=n_init, replace=False))
    candidate_idx = list(set(range(len(D_pool))) - set(selected_idx))
    
    force_rmse_history = []
    
    for step in range(n_steps):
        D_tr = [D_pool[i] for i in selected_idx]
        J_tr = [J_pool[i] for i in selected_idx]
        
        y_tr = np.zeros(len(selected_idx) * dim)
        for i, idx in enumerate(selected_idx):
            y_tr[i*dim : (i+1)*dim] = y_pool[idx]
            
        K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
        c_tr, low = cho_factor(K_tr)
        
        K_star_test = build_K_star(D_test, J_test, D_tr, J_tr, l, v)
        alpha = cho_solve((c_tr, low), y_tr)
        y_pred = K_star_test @ alpha
        
        err = (y_test - y_pred) * E_std
        idx_F_test = np.setdiff1d(np.arange(len(D_test) * dim), np.arange(0, len(D_test) * dim, dim))
        f_rmse = np.sqrt(np.mean(err[idx_F_test]**2))
        force_rmse_history.append(f_rmse)
        
        if step == n_steps - 1:
            break
            
        if strategy == "random":
            next_cand = rng.choice(candidate_idx)
        else:
            eval_sub = rng.choice(candidate_idx, size=min(60, len(candidate_idx)), replace=False)
            scores = [compute_acquisition_score(strategy, D_pool[ci], J_pool[ci], D_tr, J_tr, c_tr, l, v, nE, nF) for ci in eval_sub]
            next_cand = eval_sub[np.argmax(scores)]
            
        selected_idx.append(next_cand)
        candidate_idx.remove(next_cand)
        
    return force_rmse_history

def main():
    print("--- Step 4: Active Learning Benchmarking (Log-Det D-Optimality vs Variance) ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3
    
    c_idx, o_idx = np.where(z == 6)[0][0], np.where(z == 8)[0][0]
    distances = np.linalg.norm(R[:, c_idx, :] - R[:, o_idx, :], axis=1)
    
    gap_min, gap_max = np.mean(distances) + 0.2 * np.std(distances), np.mean(distances) + 1.2 * np.std(distances)
    in_gap = (distances >= gap_min) & (distances <= gap_max)
    
    test_frames = np.where(in_gap)[0][:20]
    pool_frames = np.where(~in_gap)[0][:200]
    
    print("Loading normalized SOAP features...")
    D_test, J_test = get_normalized_soap(z, R[test_frames], symbols, 5.0, 4, 3)
    D_pool, J_pool = get_normalized_soap(z, R[pool_frames], symbols, 5.0, 4, 3)
    
    y_test = np.zeros(len(test_frames) * dim)
    for i, idx in enumerate(test_frames):
        y_test[i*dim] = E_norm[idx]
        y_test[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
        
    y_pool = [np.concatenate([[E_norm[idx]], F_norm[idx].flatten()]) for idx in pool_frames]
    
    strategies = ["random", "max_variance", "info_gain"]
    results = {}
    
    for strat in strategies:
        print(f"\nRunning Strategy: {strat}...")
        t0 = time.time()
        results[strat] = run_active_learning(strat, D_pool, J_pool, y_pool, D_test, J_test, y_test, E_std, n_init=5, n_steps=10)
        print(f"Completed in {time.time()-t0:.1f}s. Final Force RMSE: {results[strat][-1]:.3f} kcal/mol/Å")
        
    plt.figure(figsize=(7.5, 4.5))
    steps = np.arange(5, 15)
    plt.plot(steps, results["random"], 'o--', label='Random Sampling', color='gray')
    plt.plot(steps, results["max_variance"], 's-', label='Max-Variance (Trace Heuristic)', color='C3')
    plt.plot(steps, results["info_gain"], 'd-', label='Information-Gain (Log-Det D-Optimality)', color='C0')
    plt.xlabel('Training Set Size (N)')
    plt.ylabel('Extrapolation Force RMSE (kcal/mol/Å)')
    plt.title('Active Learning Benchmark on Stretched C–O Gap')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/active_learning_gap_benchmark.png', dpi=300)
    print("\nSaved comparison plot to figures/active_learning_gap_benchmark.png")

if __name__ == "__main__":
    main()