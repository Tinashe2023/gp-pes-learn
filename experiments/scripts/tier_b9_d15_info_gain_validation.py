"""
experiments/scripts/tier_b9_d15_info_gain_validation.py

Tier B.9, D.15 & D.16: Information-Gain & Theoretical Decay Validation
1. Tier B.9: Compares predicted acquisition score against realized reduction
   in pool posterior entropy (0.5 * logdet Sigma_pool) and test error.
2. Tier D.15: Plots empirical cumulative information gain gamma_T vs T
   against the sublinear logarithmic growth envelope O(d_eff * log T).
3. Tier D.16: Plots max, 95th-percentile, and mean pool posterior variance
   vs T against the O(gamma_T / T) decay envelope.
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import energy_energy, energy_force, force_energy, force_force
from experiments.scripts.tune_soap_ethanol import get_normalized_soap

RESULTS_FILE = Path("experiments/results/tier_b9_d15_validation.json")
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


def eval_pool_uncertainty(eval_indices, D_pool, J_pool, D_tr, J_tr, c_tr, l, v, inv_Omega):
    """Computes predictive trace variance and mutual info score for every candidate in eval_indices."""
    n_atoms = D_pool[0].shape[0]
    dim = 1 + n_atoms * 3
    traces = []
    mi_scores = []

    for ci in eval_indices:
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

    return np.array(traces), np.array(mi_scores)


def main():
    print("--- Tier B.9, D.15 & D.16: Information-Gain & Theoretical Decay Validation ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]

    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3

    with open("experiments/configs/split_structural.json", "r") as f:
        split = json.load(f)

    pool_frames = split["pool"][::150][:300]
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

    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05
    inv_Omega = np.diag([1.0 / nE**2] + [1.0 / nF**2] * (dim - 1))

    rng = np.random.default_rng(42)
    n_init = 5
    n_steps = 25  # T = 1..25 queries (N = 5..30)

    selected_idx = list(rng.choice(len(D_pool), size=n_init, replace=False))
    # Fixed validation probe set of 80 pool configurations to measure global pool uncertainty
    remaining_all = list(set(range(len(D_pool))) - set(selected_idx))
    probe_idx = list(rng.choice(remaining_all, size=80, replace=False))
    candidate_idx = list(set(remaining_all) - set(probe_idx))

    predicted_ig = []
    realized_pool_mi_drop = []
    realized_pool_var_drop = []
    realized_test_rmse_drop = []

    cum_gamma_T = []
    max_var_T = []
    p95_var_T = []
    mean_var_T = []

    gamma_accum = 0.0

    for step in range(n_steps):
        D_tr = [D_pool[i] for i in selected_idx]
        J_tr = [J_pool[i] for i in selected_idx]
        y_tr = np.concatenate([y_pool_blocks[i] for i in selected_idx])

        K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
        c_tr, low = cho_factor(K_tr)

        # Current test error
        K_star_test = build_K_star(D_test, J_test, D_tr, J_tr, l, v)
        alpha = cho_solve((c_tr, low), y_tr)
        err_before = (y_test - K_star_test @ alpha) * E_std
        f_rmse_before = float(np.sqrt(np.mean(err_before[idx_F_test] ** 2)))

        # Current probe-set uncertainty
        probe_traces_before, probe_mi_before = eval_pool_uncertainty(
            probe_idx, D_pool, J_pool, D_tr, J_tr, c_tr, l, v, inv_Omega
        )
        total_probe_mi_before = float(np.sum(probe_mi_before))
        mean_probe_var_before = float(np.mean(probe_traces_before))

        max_var_T.append(float(np.max(probe_traces_before)))
        p95_var_T.append(float(np.percentile(probe_traces_before, 95)))
        mean_var_T.append(mean_probe_var_before)

        # Score candidates and select highest Log-Det MI
        eval_sub = rng.choice(candidate_idx, size=min(60, len(candidate_idx)), replace=False)
        _, cand_mi = eval_pool_uncertainty(eval_sub, D_pool, J_pool, D_tr, J_tr, c_tr, l, v, inv_Omega)
        best_local = int(np.argmax(cand_mi))
        pred_score = float(cand_mi[best_local])
        next_cand = int(eval_sub[best_local])

        gamma_accum += pred_score
        cum_gamma_T.append(float(gamma_accum))

        # Add point and measure realized drops
        selected_idx.append(next_cand)
        candidate_idx.remove(next_cand)

        D_tr_new = [D_pool[i] for i in selected_idx]
        J_tr_new = [J_pool[i] for i in selected_idx]
        y_tr_new = np.concatenate([y_pool_blocks[i] for i in selected_idx])
        K_tr_new = build_K(D_tr_new, J_tr_new, l, v, nE, nF)
        c_tr_new, low_new = cho_factor(K_tr_new)

        K_star_test_new = build_K_star(D_test, J_test, D_tr_new, J_tr_new, l, v)
        alpha_new = cho_solve((c_tr_new, low_new), y_tr_new)
        err_after = (y_test - K_star_test_new @ alpha_new) * E_std
        f_rmse_after = float(np.sqrt(np.mean(err_after[idx_F_test] ** 2)))

        probe_traces_after, probe_mi_after = eval_pool_uncertainty(
            probe_idx, D_pool, J_pool, D_tr_new, J_tr_new, c_tr_new, l, v, inv_Omega
        )

        predicted_ig.append(pred_score)
        realized_pool_mi_drop.append(total_probe_mi_before - float(np.sum(probe_mi_after)))
        realized_pool_var_drop.append(mean_probe_var_before - float(np.mean(probe_traces_after)))
        realized_test_rmse_drop.append(f_rmse_before - f_rmse_after)

        print(
            f"Step {step + 1:02d}/25 | Pred IG: {pred_score:.3f} nats | "
            f"Realized Pool MI Drop: {realized_pool_mi_drop[-1]:.3f} nats | "
            f"Max Var: {max_var_T[-1]:.5f}"
        )

    r_mi, p_mi = pearsonr(predicted_ig, realized_pool_mi_drop)
    rho_mi, _ = spearmanr(predicted_ig, realized_pool_mi_drop)
    r_var, _ = pearsonr(predicted_ig, realized_pool_var_drop)

    print("\n=== Tier B.9 Correlation Summary ===")
    print(f"Predicted IG vs. Realized Pool Entropy Reduction: Pearson r = {r_mi:.3f} (p = {p_mi:.2e}), Spearman rho = {rho_mi:.3f}")
    print(f"Predicted IG vs. Realized Mean Variance Drop:     Pearson r = {r_var:.3f}")

    summary = {
        "predicted_ig": predicted_ig,
        "realized_pool_mi_drop": realized_pool_mi_drop,
        "realized_pool_var_drop": realized_pool_var_drop,
        "realized_test_rmse_drop": realized_test_rmse_drop,
        "cum_gamma_T": cum_gamma_T,
        "max_var_T": max_var_T,
        "p95_var_T": p95_var_T,
        "mean_var_T": mean_var_T,
        "pearson_r_mi": float(r_mi),
        "spearman_rho_mi": float(rho_mi),
        "pearson_r_var": float(r_var),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    # 3-Panel Figure covering Tier B.9, Tier D.15, and Tier D.16
    T_vals = np.arange(1, n_steps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1 (Tier B.9): Scatter plot of Predicted IG vs Realized Pool Entropy Reduction
    ax = axes[0]
    ax.scatter(predicted_ig, realized_pool_mi_drop, c=T_vals, cmap="viridis", edgecolor="k", s=55)
    m, b = np.polyfit(predicted_ig, realized_pool_mi_drop, 1)
    x_line = np.linspace(min(predicted_ig), max(predicted_ig), 50)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"Linear Fit (r = {r_mi:.2f})")
    ax.set_xlabel("Predicted Acquisition Score I(y_x; f | D_t) [nats]")
    ax.set_ylabel("Realized Global Pool Entropy Reduction [nats]")
    ax.set_title("Tier B.9: Acquisition Score Validation")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # Panel 2 (Tier D.15): Empirical Information Gain Growth gamma_T vs Logarithmic Envelope
    ax = axes[1]
    ax.plot(T_vals, cum_gamma_T, "d-", color="C0", label="Empirical γ_T (Log-Det MI)")
    # Fit sublinear theoretical envelope C * log(1 + alpha * T) for visual comparison
    log_env = (cum_gamma_T[-1] / np.log(1 + n_steps)) * np.log(1 + T_vals)
    ax.plot(T_vals, log_env, "k--", label="Sublinear Envelope O(d_eff log T)")
    ax.set_xlabel("Active Learning Step (T)")
    ax.set_ylabel("Cumulative Information Gain γ_T [nats]")
    ax.set_title("Tier D.15: Sublinear Information Gain Growth")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # Panel 3 (Tier D.16): Posterior Covariance Decay vs O(gamma_T / T) Rate
    ax = axes[2]
    ax.plot(T_vals, max_var_T, "s-", color="C3", label="Max Pool Trace Variance")
    ax.plot(T_vals, p95_var_T, "^--", color="C1", label="95th Percentile Variance")
    ax.plot(T_vals, mean_var_T, "o-", color="C2", label="Mean Pool Variance")
    theory_rate = mean_var_T[0] * (np.array(cum_gamma_T) / T_vals) / (cum_gamma_T[0] / 1.0)
    ax.plot(T_vals, theory_rate, "k:", linewidth=1.8, label="Theoretical Envelope O(γ_T / T)")
    ax.set_xlabel("Active Learning Step (T)")
    ax.set_ylabel("Posterior Predictive Trace Variance")
    ax.set_title("Tier D.16: Posterior Variance Decay")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/tier_b9_d15_d16_validation.png", dpi=300)
    print("\nSaved validation figure to figures/tier_b9_d15_d16_validation.png")


if __name__ == "__main__":
    main()