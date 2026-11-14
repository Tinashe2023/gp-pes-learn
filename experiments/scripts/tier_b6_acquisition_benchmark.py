"""
experiments/scripts/tier_b6_acquisition_benchmark.py

Tier B.6 & B.7: Comprehensive 5-Strategy Multi-Seed Active Learning Benchmark
Evaluates Random, Max-Variance (Trace), Query-by-Committee (QBC), D-Optimality,
and Information-Gain (Log-Det Mutual Information) on the frozen structural split.
Logs Energy RMSE, Gap Force RMSE, Max Predictive Variance, Cumulative Info Gain,
and Wall-Clock Time across 5 seeds with checkpointing.
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

RESULTS_FILE = Path("experiments/results/tier_b6_b7_multiseed.json")
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


def compute_predictive_covariance_block(D_cand, J_cand, D_train, J_train, c_train, l, v):
    n_atoms = D_cand.shape[0]
    dim = 1 + n_atoms * 3
    K_star = build_K_star([D_cand], [J_cand], D_train, J_train, l, v)

    K_self = np.zeros((dim, dim))
    K_self[0, 0] = energy_energy(D_cand, D_cand, l, v)
    K_self[0, 1:] = energy_force(D_cand, D_cand, J_cand, l, v).flatten()
    K_self[1:, 0] = force_energy(D_cand, J_cand, D_cand, l, v).flatten()
    K_self[1:, 1:] = force_force(D_cand, J_cand, D_cand, J_cand, l, v).reshape(n_atoms * 3, n_atoms * 3)

    v_solve = cho_solve((c_train, False), K_star.T)
    Sigma_cand = K_self - K_star @ v_solve
    return Sigma_cand


def fit_qbc_committee(D_tr, J_tr, y_tr_blocks, l, v, nE, nF, rng, n_members=5):
    """Fits a 5-member bootstrap ensemble (80% subsampling) for Query-by-Committee."""
    N = len(D_tr)
    sub_size = max(3, int(0.8 * N))
    dim = 1 + D_tr[0].shape[0] * 3
    members = []

    for _ in range(n_members):
        sub_idx = rng.choice(N, size=sub_size, replace=False)
        D_sub = [D_tr[i] for i in sub_idx]
        J_sub = [J_tr[i] for i in sub_idx]
        y_sub = np.concatenate([y_tr_blocks[i] for i in sub_idx])

        K_sub = build_K(D_sub, J_sub, l, v, nE, nF)
        c_sub, low_sub = cho_factor(K_sub)
        alpha_sub = cho_solve((c_sub, low_sub), y_sub)
        members.append((D_sub, J_sub, alpha_sub, dim))
    return members


def score_qbc_candidates(eval_sub, D_pool, J_pool, members, l, v):
    """Computes force-prediction disagreement variance across the GP committee."""
    scores = []
    for ci in eval_sub:
        force_preds = []
        for D_sub, J_sub, alpha_sub, dim in members:
            K_s = build_K_star([D_pool[ci]], [J_pool[ci]], D_sub, J_sub, l, v)
            pred = K_s @ alpha_sub
            force_preds.append(pred[1:])  # Force components only
        force_preds = np.array(force_preds)  # (n_members, 27)
        # Sum of variance across force components
        disagreement = np.sum(np.var(force_preds, axis=0))
        scores.append(float(disagreement))
    return scores


def compute_single_candidate_metrics(D_cand, J_cand, D_train, J_train, c_train, l, v, nE, nF):
    """Returns trace, d_opt logdet, and mutual information gain for one candidate."""
    Sigma_cand = compute_predictive_covariance_block(D_cand, J_cand, D_train, J_train, c_train, l, v)
    dim = Sigma_cand.shape[0]
    Omega = np.diag([nE**2] + [nF**2] * (dim - 1))

    trace_val = float(np.trace(Sigma_cand))

    # D-Optimality: logdet(Sigma_cand + jitter) without observation noise weighting
    sign_d, logdet_d = np.linalg.slogdet(Sigma_cand + 1e-8 * np.eye(dim))
    d_opt_val = float(logdet_d) if sign_d > 0 else -1e10

    # Mutual Information Gain: 0.5 * logdet(I + Omega^{-1} Sigma_cand)
    inv_Omega_Sigma = np.diag(1.0 / np.diag(Omega)) @ Sigma_cand
    sign_ig, logdet_ig = np.linalg.slogdet(np.eye(dim) + inv_Omega_Sigma)
    info_gain_val = float(0.5 * logdet_ig) if sign_ig > 0 else -1e10

    return trace_val, d_opt_val, info_gain_val


def load_or_init_results(strategies):
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        for s in strategies:
            if s not in data:
                data[s] = []
        return data
    return {s: [] for s in strategies}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)


def plot_tier_b7_curves(results, n_init, n_steps):
    steps = np.arange(n_init, n_init + n_steps)
    style_map = {
        "random": ("gray", "o", "--", "Random Selection"),
        "max_variance": ("C3", "s", "-", "Max-Variance (Trace)"),
        "qbc": ("C2", "^", "-.", "Committee Disagreement (QBC)"),
        "d_optimality": ("C1", "v", ":", "D-Optimality (Unweighted)"),
        "info_gain": ("C0", "d", "-", "Info-Gain (Log-Det MI)"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics_info = [
        ("force_rmse", "Gap Force RMSE (kcal/mol/Å)", axes[0, 0]),
        ("energy_rmse", "Gap Energy RMSE (kcal/mol)", axes[0, 1]),
        ("max_pred_var", "Max Pool Predictive Trace Variance", axes[1, 0]),
        ("cum_info_gain", "Cumulative Information Gain (nats)", axes[1, 1]),
    ]

    for metric_key, ylabel, ax in metrics_info:
        for strat, (color, marker, ls, label) in style_map.items():
            runs = results.get(strat, [])
            if not runs:
                continue
            mat = np.array([r[metric_key] for r in runs])
            mean_curve = np.mean(mat, axis=0)
            std_curve = np.std(mat, axis=0)

            ax.plot(steps, mean_curve, marker=marker, linestyle=ls, color=color, label=label, markersize=5)
            ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

        ax.set_xlabel("Training Set Size (N)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if metric_key == "force_rmse":
            ax.legend(fontsize=8.5)

    plt.suptitle("Tier B.6 & B.7: Multi-Seed Active Learning Comparison on Structural Gap (Mean ± 1σ)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/tier_b6_b7_learning_curves.png", dpi=300)
    print("\nSaved 4-panel learning curves to figures/tier_b6_b7_learning_curves.png")


def print_summary_table(results):
    print("\n=== Tier B.6 Summary Table at Final Budget (N = 20) ===")
    print(f"{'Strategy':<28} {'Gap F-RMSE':<18} {'Gap E-RMSE':<18} {'Cum InfoGain':<15} {'Time/Run (s)':<12}")
    print("-" * 92)
    for strat, runs in results.items():
        if not runs:
            continue
        f_final = [r["force_rmse"][-1] for r in runs]
        e_final = [r["energy_rmse"][-1] for r in runs]
        ig_final = [r["cum_info_gain"][-1] for r in runs]
        t_total = [r["wall_time_sec"] for r in runs]
        print(
            f"{strat:<28} "
            f"{np.mean(f_final):.3f} ± {np.std(f_final):.3f}    "
            f"{np.mean(e_final):.3f} ± {np.std(e_final):.3f}    "
            f"{np.mean(ig_final):.2f} ± {np.std(ig_final):.2f}   "
            f"{np.mean(t_total):.1f}"
        )


def main():
    print("--- Tier B.6 & B.7: 5-Strategy Multi-Seed Benchmark (Frozen Structural Split) ---")
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]

    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    dim = 1 + len(z) * 3

    with open("experiments/configs/split_structural.json", "r") as f:
        split = json.load(f)

    # Fixed candidate pool and gap test set from frozen split
    pool_frames = split["pool"][:400]
    test_frames = split["test"][:50]

    print("Extracting normalized SOAP descriptors and Jacobians...")
    D_pool, J_pool = get_normalized_soap(z, R[pool_frames], symbols, 5.0, 4, 3)
    D_test, J_test = get_normalized_soap(z, R[test_frames], symbols, 5.0, 4, 3)

    y_test = np.zeros(len(test_frames) * dim)
    for i, idx in enumerate(test_frames):
        y_test[i * dim] = E_norm[idx]
        y_test[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

    y_pool_blocks = [np.concatenate([[E_norm[idx]], F_norm[idx].flatten()]) for idx in pool_frames]

    strategies = ["random", "max_variance", "qbc", "d_optimality", "info_gain"]
    results = load_or_init_results(strategies)

    n_seeds = 5
    n_init = 5
    n_steps = 16  # N = 5 up to N = 20
    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05

    idx_E_test = np.arange(0, len(test_frames) * dim, dim)
    idx_F_test = np.setdiff1d(np.arange(len(test_frames) * dim), idx_E_test)

    for seed in range(n_seeds):
        print(f"\n==================== SEED {seed + 1}/{n_seeds} ====================")
        for strategy in strategies:
            if len(results[strategy]) > seed:
                print(f"  [Skip] {strategy} already completed for Seed {seed + 1}")
                continue

            t_start = time.time()
            rng = np.random.default_rng(100 + seed)
            selected_idx = list(rng.choice(len(D_pool), size=n_init, replace=False))
            candidate_idx = list(set(range(len(D_pool))) - set(selected_idx))

            force_rmse_hist = []
            energy_rmse_hist = []
            max_var_hist = []
            cum_ig_hist = []
            cum_ig = 0.0

            for step in range(n_steps):
                D_tr = [D_pool[i] for i in selected_idx]
                J_tr = [J_pool[i] for i in selected_idx]
                y_tr_blocks = [y_pool_blocks[i] for i in selected_idx]
                y_tr = np.concatenate(y_tr_blocks)

                K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
                c_tr, low = cho_factor(K_tr)

                # Evaluate test error on structural gap
                K_star_test = build_K_star(D_test, J_test, D_tr, J_tr, l, v)
                alpha = cho_solve((c_tr, low), y_tr)
                y_pred = K_star_test @ alpha

                err = (y_test - y_pred) * E_std
                e_rmse = float(np.sqrt(np.mean(err[idx_E_test] ** 2)))
                f_rmse = float(np.sqrt(np.mean(err[idx_F_test] ** 2)))
                energy_rmse_hist.append(e_rmse)
                force_rmse_hist.append(f_rmse)

                # Evaluate candidate subset for acquisition and uncertainty tracking
                eval_sub = rng.choice(candidate_idx, size=min(60, len(candidate_idx)), replace=False)
                cand_metrics = [
                    compute_single_candidate_metrics(D_pool[ci], J_pool[ci], D_tr, J_tr, c_tr, l, v, nE, nF)
                    for ci in eval_sub
                ]
                traces = [m[0] for m in cand_metrics]
                d_opts = [m[1] for m in cand_metrics]
                info_gains = [m[2] for m in cand_metrics]

                max_var_hist.append(float(np.max(traces)))
                cum_ig_hist.append(float(cum_ig))

                if step == n_steps - 1:
                    break

                if strategy == "random":
                    chosen_local = int(rng.integers(0, len(eval_sub)))
                elif strategy == "max_variance":
                    chosen_local = int(np.argmax(traces))
                elif strategy == "d_optimality":
                    chosen_local = int(np.argmax(d_opts))
                elif strategy == "info_gain":
                    chosen_local = int(np.argmax(info_gains))
                elif strategy == "qbc":
                    members = fit_qbc_committee(D_tr, J_tr, y_tr_blocks, l, v, nE, nF, rng, n_members=5)
                    qbc_scores = score_qbc_candidates(eval_sub, D_pool, J_pool, members, l, v)
                    chosen_local = int(np.argmax(qbc_scores))

                next_cand = int(eval_sub[chosen_local])
                cum_ig += info_gains[chosen_local]
                selected_idx.append(next_cand)
                candidate_idx.remove(next_cand)

            wall_time = time.time() - t_start
            results[strategy].append(
                {
                    "force_rmse": force_rmse_hist,
                    "energy_rmse": energy_rmse_hist,
                    "max_pred_var": max_var_hist,
                    "cum_info_gain": cum_ig_hist,
                    "wall_time_sec": wall_time,
                }
            )
            save_results(results)
            print(
                f"  [Done] {strategy:<14} | Final F-RMSE: {force_rmse_hist[-1]:.3f} | "
                f"E-RMSE: {energy_rmse_hist[-1]:.3f} | Time: {wall_time:.1f}s"
            )

    print_summary_table(results)
    plot_tier_b7_curves(results, n_init, n_steps)


if __name__ == "__main__":
    main()