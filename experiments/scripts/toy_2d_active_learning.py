"""
experiments/scripts/toy_2d_active_learning.py

Step 8: 2D Toy PES Visualization
Demonstrates the active learning acquisition behavior (Random vs Max-Var vs Info-Gain)
on a 2D double-well potential to visually prove why Log-Det D-Optimality works.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

# 1. Define the 2D Asymmetric Double-Well Potential
def pes_2d(X):
    """V(x,y) = x^4 - 2x^2 + 0.3x + 2y^2"""
    x, y = X[:, 0], X[:, 1]
    E = x**4 - 2*x**2 + 0.3*x + 2*y**2
    Fx = -(4*x**3 - 4*x + 0.3)
    Fy = -4*y
    return E, np.column_stack([Fx, Fy])

# 2. Define the 2D Derivative-GP Kernel Blocks
def _rbf(Xa, Xb, l, v):
    diff = Xa[:, None, :] - Xb[None, :, :] # (Na, Nb, 2)
    K = v * np.exp(-np.sum(diff**2, axis=-1) / (2 * l**2))
    return K, diff

def cov_EE(Xa, Xb, l, v):
    K, _ = _rbf(Xa, Xb, l, v)
    return K

def cov_EF(Xa, Xb, l, v):
    K, diff = _rbf(Xa, Xb, l, v)
    return K[:, :, None] * diff / (l**2)  # Shape: (Na, Nb, 2)

def cov_FE(Xa, Xb, l, v):
    K, diff = _rbf(Xa, Xb, l, v)
    return -K[:, :, None] * diff / (l**2) # Shape: (Na, Nb, 2)

def cov_FF(Xa, Xb, l, v):
    K, diff = _rbf(Xa, Xb, l, v)
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    hess = K[:, :, None, None] * (np.eye(2)[None, None] / l**2 - outer / l**4)
    return hess # Shape: (Na, Nb, 2, 2)

def build_K(X_train, l, v, nE, nF):
    N = len(X_train)
    K_mat = np.zeros((N * 3, N * 3))
    
    KEE = cov_EE(X_train, X_train, l, v)
    KEF = cov_EF(X_train, X_train, l, v)
    KFE = cov_FE(X_train, X_train, l, v)
    KFF = cov_FF(X_train, X_train, l, v)
    
    for i in range(N):
        for j in range(N):
            r, c = i * 3, j * 3
            K_mat[r, c] = KEE[i, j]
            K_mat[r, c+1 : c+3] = KEF[i, j]
            K_mat[r+1 : r+3, c] = KFE[i, j]
            K_mat[r+1 : r+3, c+1 : c+3] = KFF[i, j]
            
    idx_E = np.arange(0, N * 3, 3)
    idx_F = np.setdiff1d(np.arange(N * 3), idx_E)
    K_mat[idx_E, idx_E] += nE**2
    K_mat[idx_F, idx_F] += nF**2
    return K_mat

def build_K_star(X_cand, X_train, l, v):
    Nc, Nt = len(X_cand), len(X_train)
    K_star = np.zeros((Nc * 3, Nt * 3))
    
    KEE = cov_EE(X_cand, X_train, l, v)
    KEF = cov_EF(X_cand, X_train, l, v)
    KFE = cov_FE(X_cand, X_train, l, v)
    KFF = cov_FF(X_cand, X_train, l, v)
    
    for i in range(Nc):
        for j in range(Nt):
            K_star[i*3, j*3] = KEE[i, j]
            K_star[i*3, j*3+1 : j*3+3] = KEF[i, j]
            K_star[i*3+1 : i*3+3, j*3] = KFE[i, j]
            K_star[i*3+1 : i*3+3, j*3+1 : j*3+3] = KFF[i, j]
            
    return K_star

def compute_acquisition(strategy, x_cand, X_train, c_train, l, v, nE, nF):
    K_star = build_K_star(np.array([x_cand]), X_train, l, v)
    
    K_self = np.zeros((3, 3))
    K_self[0, 0] = v
    K_self[1:, 1:] = np.eye(2) * (v / l**2)
    
    v_solve = cho_solve((c_train, False), K_star.T)
    Sigma_cand = K_self - K_star @ v_solve
    
    if strategy == "max_variance":
        return np.trace(Sigma_cand)
    elif strategy == "info_gain":
        Omega = np.diag([nE**2, nF**2, nF**2])
        sign, logdet = np.linalg.slogdet(Sigma_cand + Omega)
        return logdet if sign > 0 else -1e10

def main():
    print("--- Step 8: Running 2D Toy PES Active Learning ---")
    l, v, nE, nF = 0.5, 10.0, 0.01, 0.05
    
    # Create candidate pool grid
    xx, yy = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-1.5, 1.5, 25))
    X_pool = np.column_stack([xx.ravel(), yy.ravel()])
    E_pool, _ = pes_2d(X_pool)
    
    # Init with 1 random point
    n_queries = 10
    rng = np.random.default_rng(42)
    init_idx = rng.choice(len(X_pool))
    
    results = {}
    strategies = ["random", "max_variance", "info_gain"]
    
    for strat in strategies:
        print(f"Sampling via {strat}...")
        selected_idx = [init_idx]
        candidate_idx = list(set(range(len(X_pool))) - set([init_idx]))
        
        for step in range(n_queries):
            X_tr = X_pool[selected_idx]
            K_tr = build_K(X_tr, l, v, nE, nF)
            c_tr, _ = cho_factor(K_tr)
            
            if strat == "random":
                next_cand = rng.choice(candidate_idx)
            else:
                scores = [compute_acquisition(strat, X_pool[ci], X_tr, c_tr, l, v, nE, nF) for ci in candidate_idx]
                next_cand = candidate_idx[np.argmax(scores)]
                
            selected_idx.append(next_cand)
            candidate_idx.remove(next_cand)
            
        results[strat] = X_pool[selected_idx]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles = {"random": "Random Sampling", "max_variance": "Max-Variance (Trace)", "info_gain": "Info-Gain (Log-Det D-Opt)"}
    
    for ax, strat in zip(axes, strategies):
        # Background contours
        contour = ax.contourf(xx, yy, E_pool.reshape(25, 25), levels=20, cmap='viridis', alpha=0.8)
        ax.contour(xx, yy, E_pool.reshape(25, 25), levels=20, colors='k', linewidths=0.5, alpha=0.3)
        
        # Plot queries
        pts = results[strat]
        ax.plot(pts[0, 0], pts[0, 1], 'k*', markersize=12, label="Initial Point")
        ax.plot(pts[1:, 0], pts[1:, 1], 'ro', markersize=6, markeredgecolor='white', label="AL Queries")
        
        # Draw path
        ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1, alpha=0.5)
        
        ax.set_title(titles[strat])
        ax.set_xlabel('Coordinate X')
        if ax == axes[0]:
            ax.set_ylabel('Coordinate Y')
            ax.legend(loc='upper left')

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/toy_2d_pes_comparison.png", dpi=300)
    print("\nSaved visualization to figures/toy_2d_pes_comparison.png")

if __name__ == "__main__":
    main()