"""
experiments/scripts/analytic_1d_benchmark.py

Tier A.2: Quantitative 1D Analytic Benchmark.
Compares Energy-Only GP vs Energy+Force GP, and 
Random Sampling vs Information Gain on a 1D function with a central gap.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

def f_1d(x):
    """1D multi-frequency function: f(x) = sin(x) + 0.5*cos(3x)"""
    E = np.sin(x) + 0.5 * np.cos(3 * x)
    F = -(np.cos(x) - 1.5 * np.sin(3 * x)) # Force = -dE/dx
    return E, F

def rbf_1d(xa, xb, l, v):
    diff = xa[:, None] - xb[None, :]
    K = v * np.exp(-0.5 * (diff / l)**2)
    return K, diff

def build_K_1D_ef(x_train, l, v, nE, nF):
    N = len(x_train)
    K = np.zeros((N * 2, N * 2))
    K_EE, diff = rbf_1d(x_train, x_train, l, v)
    
    K_EF = K_EE * diff / l**2
    K_FE = -K_EE * diff / l**2
    K_FF = K_EE * (1.0 / l**2 - (diff / l**2)**2)
    
    for i in range(N):
        for j in range(N):
            K[i*2, j*2] = K_EE[i, j]
            K[i*2, j*2+1] = K_EF[i, j]
            K[i*2+1, j*2] = K_FE[i, j]
            K[i*2+1, j*2+1] = K_FF[i, j]
            
    idx_E = np.arange(0, N * 2, 2)
    idx_F = np.arange(1, N * 2, 2)
    K[idx_E, idx_E] += nE**2
    K[idx_F, idx_F] += nF**2
    return K

def build_K_star_1D_ef(x_test, x_train, l, v):
    Nt, Ntr = len(x_test), len(x_train)
    K_star = np.zeros((Nt * 2, Ntr * 2))
    K_EE, diff = rbf_1d(x_test, x_train, l, v)
    
    K_EF = K_EE * diff / l**2
    K_FE = -K_EE * diff / l**2
    K_FF = K_EE * (1.0 / l**2 - (diff / l**2)**2)
    
    for i in range(Nt):
        for j in range(Ntr):
            K_star[i*2, j*2] = K_EE[i, j]
            K_star[i*2, j*2+1] = K_EF[i, j]
            K_star[i*2+1, j*2] = K_FE[i, j]
            K_star[i*2+1, j*2+1] = K_FF[i, j]
    return K_star

def compute_info_gain(x_cand, x_train, c_tr, l, v, nE, nF):
    K_star = build_K_star_1D_ef(np.array([x_cand]), x_train, l, v)
    K_self = np.diag([v, v/l**2])
    v_solve = cho_solve((c_tr, False), K_star.T)
    Sigma = K_self - K_star @ v_solve
    Omega = np.diag([nE**2, nF**2])
    _, logdet = np.linalg.slogdet(Sigma + Omega)
    return logdet

def main():
    print("--- Tier A.2: 1D Analytic Benchmark (Gap Extrapolation) ---")
    l, v, nE, nF = 0.8, 2.0, 0.01, 0.05
    
    # 1. Define the domain and the gap
    x_dense = np.linspace(-5, 5, 200)
    E_dense, F_dense = f_1d(x_dense)
    gap_mask = (x_dense > -1.5) & (x_dense < 1.5)
    
    x_pool = x_dense[~gap_mask] # Candidate pool outside gap
    x_test = x_dense[gap_mask]  # Test set inside gap
    E_test, F_test = f_1d(x_test)
    y_test = np.empty(len(x_test) * 2)
    y_test[0::2], y_test[1::2] = E_test, F_test
    
    # Initial seeds (2 points on the far left, 2 on the far right)
    init_idx = [0, 5, len(x_pool)-6, len(x_pool)-1]
    
    results = {"random": [], "info_gain": []}
    strategies = ["random", "info_gain"]
    n_queries = 15
    rng = np.random.default_rng(42)
    
    for strat in strategies:
        print(f"\nRunning {strat}...")
        selected_idx = list(init_idx)
        candidate_idx = list(set(range(len(x_pool))) - set(init_idx))
        
        force_rmse_hist = []
        
        for step in range(n_queries):
            x_tr = x_pool[selected_idx]
            E_tr, F_tr = f_1d(x_tr)
            y_tr = np.empty(len(x_tr) * 2)
            y_tr[0::2], y_tr[1::2] = E_tr, F_tr
            
            K_tr = build_K_1D_ef(x_tr, l, v, nE, nF)
            c_tr, low = cho_factor(K_tr)
            
            # Predict in gap
            K_star = build_K_star_1D_ef(x_test, x_tr, l, v)
            alpha = cho_solve((c_tr, low), y_tr)
            y_pred = K_star @ alpha
            
            f_rmse = np.sqrt(np.mean((y_test[1::2] - y_pred[1::2])**2))
            force_rmse_hist.append(f_rmse)
            
            if step == n_queries - 1:
                break
                
            if strat == "random":
                next_c = rng.choice(candidate_idx)
            else:
                scores = [compute_info_gain(x_pool[ci], x_tr, c_tr, l, v, nE, nF) for ci in candidate_idx]
                next_c = candidate_idx[np.argmax(scores)]
                
            selected_idx.append(next_c)
            candidate_idx.remove(next_c)
            
        results[strat] = force_rmse_hist
        print(f"Final Gap Force RMSE ({strat}): {force_rmse_hist[-1]:.4f}")

    # Plotting
    plt.figure(figsize=(7, 4.5))
    steps = np.arange(4, 4 + n_queries)
    plt.plot(steps, results["random"], 'o--', color='gray', label='Random Sampling')
    plt.plot(steps, results["info_gain"], 'd-', color='C0', label='Info-Gain (Log-Det D-Opt)')
    
    plt.xlabel('Number of Training Points (N)')
    plt.ylabel('Extrapolation Gap Force RMSE')
    plt.title('1D Analytic PES: Active Learning across a Gap')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/analytic_1d_benchmark.png", dpi=300)
    print("\nSaved plot to figures/analytic_1d_benchmark.png")

if __name__ == "__main__":
    main()