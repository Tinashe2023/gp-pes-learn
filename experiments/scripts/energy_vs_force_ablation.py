"""
experiments/scripts/energy_vs_force_ablation.py

Tier C.11: Energy-Only vs. Energy-Force Ablation
Demonstrates the sample efficiency gained by including derivative observations.
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import energy_energy
from experiments.scripts.tune_soap_ethanol import get_normalized_soap, build_K, build_K_star

def build_K_energy_only(D_list, lengthscale, variance, noise_E):
    """Builds the N x N covariance matrix using ONLY energy observations."""
    N = len(D_list)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            val = energy_energy(D_list[i], D_list[j], lengthscale, variance)
            K[i, j] = val
            if i != j:
                K[j, i] = val
    np.fill_diagonal(K, K.diagonal() + noise_E**2)
    return K

def build_K_star_energy_only(D_test, D_train, lengthscale, variance):
    """Builds the N_test x N_train covariance matrix for energy predictions."""
    N_test, N_train = len(D_test), len(D_train)
    K_star = np.zeros((N_test, N_train))
    for i in range(N_test):
        for j in range(N_train):
            K_star[i, j] = energy_energy(D_test[i], D_train[j], lengthscale, variance)
    return K_star

def main():
    print("--- Tier C.11: Energy-Only vs. Energy-Force GP Ablation ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3
    
    # Load Frozen Random Split
    with open("experiments/configs/split_random.json", "r") as f:
        split = json.load(f)
        
    test_idx = split["test"][:100]
    D_test, J_test = get_normalized_soap(z, R[test_idx], symbols, 5.0, 4, 3)
    
    y_test_full = np.zeros(len(test_idx) * dim)
    y_test_energy = np.zeros(len(test_idx))
    for i, idx in enumerate(test_idx):
        y_test_energy[i] = E_norm[idx]
        y_test_full[i*dim] = E_norm[idx]
        y_test_full[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()

    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05
    N_range = [5, 10, 15, 20, 25]
    
    results = {"E_only_rmse": [], "EF_rmse": []}
    
    for N in N_range:
        print(f"\nTraining on N = {N} frames...")
        train_idx = split["pool"][:N]
        D_train, J_train = get_normalized_soap(z, R[train_idx], symbols, 5.0, 4, 3)
        
        # --- Energy-Only GP ---
        y_train_energy = np.array([E_norm[idx] for idx in train_idx])
        K_tr_E = build_K_energy_only(D_train, l, v, nE)
        c_tr_E, low_E = cho_factor(K_tr_E)
        K_star_E = build_K_star_energy_only(D_test, D_train, l, v)
        alpha_E = cho_solve((c_tr_E, low_E), y_train_energy)
        y_pred_energy = K_star_E @ alpha_E
        
        e_rmse_only = np.sqrt(np.mean(((y_test_energy - y_pred_energy) * E_std)**2))
        results["E_only_rmse"].append(e_rmse_only)
        
        # --- Energy-Force GP ---
        y_train_full = np.zeros(N * dim)
        for i, idx in enumerate(train_idx):
            y_train_full[i*dim] = E_norm[idx]
            y_train_full[i*dim+1 : (i+1)*dim] = F_norm[idx].flatten()
            
        K_tr_EF = build_K(D_train, J_train, l, v, nE, nF)
        c_tr_EF, low_EF = cho_factor(K_tr_EF)
        K_star_EF = build_K_star(D_test, J_test, D_train, J_train, l, v)
        alpha_EF = cho_solve((c_tr_EF, low_EF), y_train_full)
        y_pred_full = K_star_EF @ alpha_EF
        
        idx_E = np.arange(0, len(y_test_full), dim)
        e_rmse_EF = np.sqrt(np.mean(((y_test_full[idx_E] - y_pred_full[idx_E]) * E_std)**2))
        results["EF_rmse"].append(e_rmse_EF)
        
        print(f"  Energy-Only RMSE:  {e_rmse_only:.3f} kcal/mol")
        print(f"  Energy-Force RMSE: {e_rmse_EF:.3f} kcal/mol")

    # Plotting
    plt.figure(figsize=(7, 4.5))
    plt.plot(N_range, results["E_only_rmse"], 'o--', color='gray', label='Energy-Only GP')
    plt.plot(N_range, results["EF_rmse"], 's-', color='C3', label='Energy-Force GP')
    plt.xlabel('Number of Training Frames (N)')
    plt.ylabel('Test Energy RMSE (kcal/mol)')
    plt.title('Sample Efficiency: Energy-Only vs. Energy-Force GP')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/energy_vs_force_ablation.png", dpi=300)
    print("\nSaved plot to figures/energy_vs_force_ablation.png")

if __name__ == "__main__":
    main()