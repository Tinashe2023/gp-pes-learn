"""
experiments/scripts/tier_c14_robustness_and_malonaldehyde.py

Tier C.14: Multi-Split & Multi-Molecule Robustness Benchmark
1. Evaluates Random, Max-Variance (Trace), and Info-Gain (Log-Det MI) across
   all three frozen Ethanol splits (Random, Contiguous, Structural) over 3 seeds.
2. Evaluates cross-molecule generalization on rMD17 Malonaldehyde (C3H4O2, 9 atoms)
   under an intramolecular O-H distance gap (testing proton-transfer regime)
   using the exact frozen hyperparameters (l=1.5, v=1.0, nE=0.01, nF=0.05).
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

RESULTS_FILE = Path("experiments/results/tier_c14_robustness.json")
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


def score_candidates(eval_sub, D_pool, J_pool, D_tr, J_tr, c_tr, l, v, inv_Omega):
    n_atoms = D_pool[0].shape[0]
    dim = 1 + n_atoms * 3
    traces, mi_scores = [], []

    for ci in eval_sub:
        D_c, J_c = D_pool[ci], J_pool[ci]
        K_star = build_K_star([D_c], [J_c], D_tr, J_tr, l, v)

        K_self = np.zeros((dim, dim))
        K_self[0, 0] = energy_energy(D_c, D_c, l, v)
        K_self[0, 1:] = energy_force(D_c, D_c, J_c, l, v).flatten()
        K_self[1:, 0] = force_energy(D_c, J_c, D_c, l, v).flatten()
        K_self[1:, 1:] = force_force(D_c, J_c, D_c, J_c, l, v).reshape(n_atoms * 3, n_atoms * 3)

        v_solve = cho_solve((c_tr, False), K_star.T)
        Sigma = K_self - K_star @ v_solve

        traces.append(float(np.trace(Sigma)))
        sign, logdet = np.linalg.slogdet(np.eye(dim) + inv_Omega @ Sigma)
        mi_scores.append(float(0.5 * logdet) if sign > 0 else -1e10)

    return traces, mi_scores


def run_benchmark_regime(name, z, R, E, F, pool_idx, test_idx, symbols, n_seeds=3, n_init=5, n_steps=16):
    print(f"\n=== Regime: {name} (Pool={len(pool_idx)}, Test={len(test_idx)}) ===")
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    dim = 1 + len(z) * 3

    D_pool, J_pool = get_normalized_soap(z, R[pool_idx], symbols, 5.0, 4, 3)
    D_test, J_test = get_normalized_soap(z, R[test_idx], symbols, 5.0, 4, 3)

    y_test = np.zeros(len(test_idx) * dim)
    for i, idx in enumerate(test_idx):
        y_test[i * dim] = E_norm[idx]
        y_test[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

    y_pool_blocks = [np.concatenate([[E_norm[idx]], F_norm[idx].flatten()]) for idx in pool_idx]
    idx_E_test = np.arange(0, len(test_idx) * dim, dim)
    idx_F_test = np.setdiff1d(np.arange(len(test_idx) * dim), idx_E_test)

    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05
    inv_Omega = np.diag([1.0 / nE**2] + [1.0 / nF**2] * (dim - 1))

    strategies = ["random", "max_variance", "info_gain"]
    regime_res = {s: [] for s in strategies}

    for seed in range(n_seeds):
        for strat in strategies:
            rng = np.random.default_rng(300 + seed)
            selected = list(rng.choice(len(D_pool), size=n_init, replace=False))
            candidates = list(set(range(len(D_pool))) - set(selected))

            f_hist, e_hist = [], []
            for step in range(n_steps):
                D_tr = [D_pool[i] for i in selected]
                J_tr = [J_pool[i] for i in selected]
                y_tr = np.concatenate([y_pool_blocks[i] for i in selected])

                K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
                c_tr, low = cho_factor(K_tr)
                K_star = build_K_star(D_test, J_test, D_tr, J_tr, l, v)
                alpha = cho_solve((c_tr, low), y_tr)
                err = (y_test - K_star @ alpha) * E_std

                e_hist.append(float(np.sqrt(np.mean(err[idx_E_test] ** 2))))
                f_hist.append(float(np.sqrt(np.mean(err[idx_F_test] ** 2))))

                if step == n_steps - 1:
                    break

                eval_sub = rng.choice(candidates, size=min(60, len(candidates)), replace=False)
                if strat == "random":
                    next_c = int(rng.choice(eval_sub))
                else:
                    traces, mis = score_candidates(eval_sub, D_pool, J_pool, D_tr, J_tr, c_tr, l, v, inv_Omega)
                    chosen = int(np.argmax(traces)) if strat == "max_variance" else int(np.argmax(mis))
                    next_c = int(eval_sub[chosen])

                selected.append(next_c)
                candidates.remove(next_c)

            regime_res[strat].append({"force_rmse": f_hist, "energy_rmse": e_hist})

    for strat in strategies:
        f_end = [r["force_rmse"][-1] for r in regime_res[strat]]
        e_end = [r["energy_rmse"][-1] for r in regime_res[strat]]
        print(
            f"  {strat:<15} | Final (N=20) F-RMSE: {np.mean(f_end):.3f} ± {np.std(f_end):.3f} kcal/mol/Å | "
            f"E-RMSE: {np.mean(e_end):.3f} ± {np.std(e_end):.3f} kcal/mol"
        )
    return regime_res


def plot_c14_summary(all_results, n_init=5, n_steps=16):
    steps = np.arange(n_init, n_init + n_steps)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("ethanol_random", "Ethanol: Random Split (Interpolation)", axes[0, 0]),
        ("ethanol_contiguous", "Ethanol: Contiguous Trajectory Split", axes[0, 1]),
        ("ethanol_structural", "Ethanol: Structural C-O Gap Split", axes[1, 0]),
        ("malonaldehyde_structural", "Malonaldehyde: Structural O-H Gap Split", axes[1, 1]),
    ]
    styles = {
        "random": ("gray", "o", "--", "Random Sampling"),
        "max_variance": ("C3", "s", "-", "Max-Variance (Trace)"),
        "info_gain": ("C0", "d", "-", "Info-Gain (Log-Det MI)"),
    }

    for key, title, ax in panels:
        res = all_results[key]
        for strat, (color, marker, ls, label) in styles.items():
            mat = np.array([r["force_rmse"] for r in res[strat]])
            mean_c = np.mean(mat, axis=0)
            std_c = np.std(mat, axis=0)
            ax.plot(steps, mean_c, marker=marker, linestyle=ls, color=color, label=label, markersize=4.5)
            ax.fill_between(steps, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Training Set Size (N)")
        ax.set_ylabel("Force RMSE (kcal/mol/Å)")
        ax.grid(True, alpha=0.4)
        if key == "ethanol_random":
            ax.legend(fontsize=8.5)

    plt.suptitle("Tier C.14: Multi-Split & Cross-Molecule Robustness (Mean ± 1σ)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/tier_c14_multi_split_and_molecule.png", dpi=300)
    print("\nSaved Tier C.14 robustness plot to figures/tier_c14_multi_split_and_molecule.png")


def main():
    print("--- Tier C.14: Multi-Split & Multi-Molecule Robustness ---")
    eth = load_rmd17("ethanol")
    symbols = ["C", "O", "H"]
    all_results = {}

    # 1. Ethanol: Random Split
    with open("experiments/configs/split_random.json", "r") as f:
        s_rand = json.load(f)
    all_results["ethanol_random"] = run_benchmark_regime(
        "Ethanol Random Split",
        eth["z"], eth["R"], eth["E"], eth["F"],
        s_rand["pool"][::150][:350], s_rand["test"][::200][:50], symbols
    )

    # 2. Ethanol: Contiguous Trajectory Split
    with open("experiments/configs/split_contiguous.json", "r") as f:
        s_cont = json.load(f)
    all_results["ethanol_contiguous"] = run_benchmark_regime(
        "Ethanol Contiguous Split",
        eth["z"], eth["R"], eth["E"], eth["F"],
        s_cont["pool"][::150][:350], s_cont["test"][::200][:50], symbols
    )

    # 3. Ethanol: Structural C-O Bond Gap Split (stride-subsampled)
    with open("experiments/configs/split_structural.json", "r") as f:
        s_struc = json.load(f)
    all_results["ethanol_structural"] = run_benchmark_regime(
        "Ethanol Structural Split",
        eth["z"], eth["R"], eth["E"], eth["F"],
        s_struc["pool"][::150][:350], s_struc["test"][::200][:50], symbols
    )

    # 4. Second Molecule: rMD17 Malonaldehyde (C3H4O2, 9 atoms) under O-H bond gap
    mal = load_rmd17("malonaldehyde")
    z_m, R_m = mal["z"], mal["R"]
    o_indices = np.where(z_m == 8)[0]
    h_indices = np.where(z_m == 1)[0]
    # Minimum O-H distance per frame (captures intramolecular proton transfer)
    oh_dists = np.min(
        [np.linalg.norm(R_m[:, o_i, :] - R_m[:, h_i, :], axis=1) for o_i in o_indices for h_i in h_indices],
        axis=0,
    )
    g_min = np.mean(oh_dists) + 0.2 * np.std(oh_dists)
    g_max = np.mean(oh_dists) + 1.2 * np.std(oh_dists)
    in_gap_m = (oh_dists >= g_min) & (oh_dists <= g_max)
    mal_pool = np.where(~in_gap_m)[0][::150][:350].tolist()
    mal_test = np.where(in_gap_m)[0][::200][:50].tolist()

    all_results["malonaldehyde_structural"] = run_benchmark_regime(
        "Malonaldehyde Structural O-H Gap",
        z_m, R_m, mal["E"], mal["F"],
        mal_pool, mal_test, symbols
    )

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4)

    plot_c14_summary(all_results)


if __name__ == "__main__":
    main()