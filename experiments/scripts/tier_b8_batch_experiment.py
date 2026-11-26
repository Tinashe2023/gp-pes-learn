"""
experiments/scripts/tier_b8_batch_experiment.py

Tier B.8: Batch-Size Experiment (B in {1, 5, 10, 25})
Evaluates greedy conditional Log-Det Information-Gain batching up to a fixed
total query budget (+25 frames from N_init=5 -> N_final=30) across 3 seeds.
Measures:
  1. Final Gap Force & Energy RMSE at equal query budgets
  2. Total wall-clock time (showing retraining speedup of batching)
  3. Intra-batch diversity (mean pairwise SOAP distance) & redundancy
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import energy_energy, energy_force, force_energy, force_force
from experiments.scripts.tune_soap_ethanol import get_normalized_soap

RESULTS_FILE = Path("experiments/results/tier_b8_batch_results.json")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(exist_ok=True)


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
            K_FF = force_force(Da, Ja, Db, Jb, lengthscale, variance).reshape(n_atoms * 3, n_atoms * 3)

            r, c = i * dim, j * dim
            K[r, c] = K_EE
            K[r, c + 1 : c + dim] = K_EF
            K[r + 1 : r + dim, c] = K_FE
            K[r + 1 : r + dim, c + 1 : c + dim] = K_FF

            if i != j:
                K[c, r] = K_EE
                K[c, r + 1 : r + dim] = K_FE
                K[c + 1 : c + dim, r] = K_EF
                K[c + 1 : c + dim, r + 1 : r + dim] = K_FF.T

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

            K_star[i * dim, j * dim] = energy_energy(Da, Db, lengthscale, variance)
            K_star[i * dim, j * dim + 1 : (j + 1) * dim] = energy_force(Da, Db, Jb, lengthscale, variance).flatten()
            K_star[i * dim + 1 : (i + 1) * dim, j * dim] = force_energy(Da, Ja, Db, lengthscale, variance).flatten()
            K_star[i * dim + 1 : (i + 1) * dim, j * dim + 1 : (j + 1) * dim] = force_force(
                Da, Ja, Db, Jb, lengthscale, variance
            ).reshape(n_atoms * 3, -1)
    return K_star


def select_greedy_batch(eval_sub, batch_size, D_pool, J_pool, D_tr, J_tr, l, v, nE, nF):
    """
    Greedy submodular batch selection using posterior covariance conditioning.
    After picking each item in the batch, updates the conditioning set (without
    needing labels, since GP posterior covariance depends only on inputs X!)
    so subsequent items in the same batch avoid redundancy.
    """
    selected_in_batch = []
    remaining = list(eval_sub)

    # Working conditioning set starts as current labeled training inputs
    D_cond = list(D_tr)
    J_cond = list(J_tr)
    n_atoms = D_pool[0].shape[0]
    dim = 1 + n_atoms * 3
    Omega = np.diag([nE**2] + [nF**2] * (dim - 1))
    inv_Omega = np.diag(1.0 / np.diag(Omega))

    for _ in range(batch_size):
        K_cond = build_K(D_cond, J_cond, l, v, nE, nF)
        c_cond, _ = cho_factor(K_cond)

        best_score = -1e12
        best_ci = remaining[0]

        for ci in remaining:
            D_c, J_c = D_pool[ci], J_pool[ci]
            K_star = build_K_star([D_c], [J_c], D_cond, J_cond, l, v)

            K_self = np.zeros((dim, dim))
            K_self[0, 0] = energy_energy(D_c, D_c, l, v)
            K_self[0, 1:] = energy_force(D_c, D_c, J_c, l, v).flatten()
            K_self[1:, 0] = force_energy(D_c, J_c, D_c, l, v).flatten()
            K_self[1:, 1:] = force_force(D_c, J_c, D_c, J_c, l, v).reshape(n_atoms * 3, n_atoms * 3)

            v_solve = cho_solve((c_cond, False), K_star.T)
            Sigma_cand = K_self - K_star @ v_solve

            sign, logdet = np.linalg.slogdet(np.eye(dim) + inv_Omega @ Sigma_cand)
            score = float(0.5 * logdet) if sign > 0 else -1e10
            if score > best_score:
                best_score = score
                best_ci = ci

        selected_in_batch.append(best_ci)
        remaining.remove(best_ci)
        D_cond.append(D_pool[best_ci])
        J_cond.append(J_pool[best_ci])

    return selected_in_batch


def compute_batch_diversity(batch_indices, D_pool):
    """Mean pairwise L2 distance between flattened normalized SOAP descriptors in a batch."""
    if len(batch_indices) <= 1:
        return 0.0
    vecs = np.array([D_pool[i].flatten() for i in batch_indices])
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.mean(dists))


def plot_batch_results(results, batch_sizes):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Learning curve (Force RMSE vs N)
    ax = axes[0]
    colors = {1: "C0", 5: "C2", 10: "C1", 25: "C3"}
    markers = {1: "d", 5: "s", 10: "^", 25: "o"}
    for B in batch_sizes:
        runs = results[str(B)]
        N_vals = runs[0]["N_history"]
        f_mat = np.array([r["force_rmse"] for r in runs])
        mean_f = np.mean(f_mat, axis=0)
        std_f = np.std(f_mat, axis=0)
        ax.plot(N_vals, mean_f, marker=markers[B], color=colors[B], label=f"Batch B = {B}")
        ax.fill_between(N_vals, mean_f - std_f, mean_f + std_f, color=colors[B], alpha=0.15)
    ax.set_xlabel("Training Set Size (N)")
    ax.set_ylabel("Gap Force RMSE (kcal/mol/Å)")
    ax.set_title("(a) Accuracy vs. Query Budget")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # Panel 2: Wall-clock runtime comparison
    ax = axes[1]
    mean_times = [np.mean([r["wall_time_sec"] for r in results[str(B)]]) for B in batch_sizes]
    std_times = [np.std([r["wall_time_sec"] for r in results[str(B)]]) for B in batch_sizes]
    ax.bar([f"B={B}" for B in batch_sizes], mean_times, yerr=std_times, capsize=4, color=[colors[B] for B in batch_sizes], alpha=0.85)
    ax.set_ylabel("Total Wall-Clock Time (s)")
    ax.set_title("(b) Retraining & Acquisition Runtime")
    ax.grid(True, axis="y", alpha=0.4)

    # Panel 3: Intra-batch diversity (for B > 1)
    ax = axes[2]
    multi_B = [B for B in batch_sizes if B > 1]
    mean_div = [np.mean([r["mean_batch_diversity"] for r in results[str(B)]]) for B in multi_B]
    std_div = [np.std([r["mean_batch_diversity"] for r in results[str(B)]]) for B in multi_B]
    ax.bar([f"B={B}" for B in multi_B], mean_div, yerr=std_div, capsize=4, color=[colors[B] for B in multi_B], alpha=0.85)
    ax.set_ylabel("Mean Pairwise SOAP Distance")
    ax.set_title("(c) Intra-Batch Structural Diversity")
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/tier_b8_batch_analysis.png", dpi=300)
    print("\nSaved batch analysis plot to figures/tier_b8_batch_analysis.png")


def main():
    print("--- Tier B.8: Batch-Size Experiment (B in {1, 5, 10, 25}) ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]

    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3

    with open("experiments/configs/split_structural.json", "r") as f:
        split = json.load(f)

    # Stride-subsampled pool so candidate configurations span the full trajectory
    pool_frames = split["pool"][::150][:400]
    test_frames = split["test"][::200][:50]

    print("Extracting normalized SOAP descriptors and Jacobians...")
    D_pool, J_pool = get_normalized_soap(z, R[pool_frames], symbols, 5.0, 4, 3)
    D_test, J_test = get_normalized_soap(z, R[test_frames], symbols, 5.0, 4, 3)

    y_test = np.zeros(len(test_frames) * dim)
    for i, idx in enumerate(test_frames):
        y_test[i * dim] = E_norm[idx]
        y_test[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

    y_pool_blocks = [np.concatenate([[E_norm[idx]], F_norm[idx].flatten()]) for idx in pool_frames]

    idx_E_test = np.arange(0, len(test_frames) * dim, dim)
    idx_F_test = np.setdiff1d(np.arange(len(test_frames) * dim), idx_E_test)

    batch_sizes = [1, 5, 10, 25]
    n_seeds = 3
    n_init = 5
    total_new_queries = 25  # Final N = 30
    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05

    results = {str(B): [] for B in batch_sizes}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)

    for seed in range(n_seeds):
        print(f"\n=== Seed {seed + 1}/{n_seeds} ===")
        for B in batch_sizes:
            if len(results[str(B)]) > seed:
                print(f"  [Skip] B={B} already completed for Seed {seed + 1}")
                continue

            t0 = time.time()
            rng = np.random.default_rng(200 + seed)
            selected_idx = list(rng.choice(len(D_pool), size=n_init, replace=False))
            candidate_idx = list(set(range(len(D_pool))) - set(selected_idx))

            N_hist = []
            f_rmse_hist = []
            e_rmse_hist = []
            batch_diversities = []

            queries_added = 0
            while True:
                # Fit GP and evaluate current model
                D_tr = [D_pool[i] for i in selected_idx]
                J_tr = [J_pool[i] for i in selected_idx]
                y_tr = np.concatenate([y_pool_blocks[i] for i in selected_idx])

                K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
                c_tr, low = cho_factor(K_tr)
                K_star_test = build_K_star(D_test, J_test, D_tr, J_tr, l, v)
                alpha = cho_solve((c_tr, low), y_tr)
                y_pred = K_star_test @ alpha

                err = (y_test - y_pred) * E_std
                e_rmse = float(np.sqrt(np.mean(err[idx_E_test] ** 2)))
                f_rmse = float(np.sqrt(np.mean(err[idx_F_test] ** 2)))

                N_hist.append(len(selected_idx))
                e_rmse_hist.append(e_rmse)
                f_rmse_hist.append(f_rmse)

                if queries_added >= total_new_queries:
                    break

                curr_b = min(B, total_new_queries - queries_added)
                eval_sub = rng.choice(candidate_idx, size=min(60, len(candidate_idx)), replace=False)

                batch_chosen = select_greedy_batch(
                    eval_sub, curr_b, D_pool, J_pool, D_tr, J_tr, l, v, nE, nF
                )
                if curr_b > 1:
                    batch_diversities.append(compute_batch_diversity(batch_chosen, D_pool))

                for ci in batch_chosen:
                    selected_idx.append(int(ci))
                    candidate_idx.remove(int(ci))
                queries_added += curr_b

            wall_time = time.time() - t0
            mean_div = float(np.mean(batch_diversities)) if batch_diversities else 0.0

            results[str(B)].append(
                {
                    "N_history": N_hist,
                    "force_rmse": f_rmse_hist,
                    "energy_rmse": e_rmse_hist,
                    "mean_batch_diversity": mean_div,
                    "wall_time_sec": wall_time,
                }
            )
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)

            print(
                f"  [Done] B={B:<2} | Final (N=30) F-RMSE: {f_rmse_hist[-1]:.3f} | "
                f"E-RMSE: {e_rmse_hist[-1]:.3f} | Diversity: {mean_div:.3f} | Time: {wall_time:.1f}s"
            )

    print("\n=== Tier B.8 Summary Table (Final N = 30) ===")
    print(f"{'Batch Size':<12} {'Final F-RMSE':<18} {'Final E-RMSE':<18} {'Batch Diversity':<18} {'Wall Time (s)':<14}")
    print("-" * 80)
    for B in batch_sizes:
        runs = results[str(B)]
        f_end = [r["force_rmse"][-1] for r in runs]
        e_end = [r["energy_rmse"][-1] for r in runs]
        divs = [r["mean_batch_diversity"] for r in runs]
        tms = [r["wall_time_sec"] for r in runs]
        print(
            f"B = {B:<8} "
            f"{np.mean(f_end):.3f} ± {np.std(f_end):.3f}    "
            f"{np.mean(e_end):.3f} ± {np.std(e_end):.3f}    "
            f"{np.mean(divs):.3f} ± {np.std(divs):.3f}    "
            f"{np.mean(tms):.1f} ± {np.std(tms):.1f}"
        )

    plot_batch_results(results, batch_sizes)


if __name__ == "__main__":
    main()
