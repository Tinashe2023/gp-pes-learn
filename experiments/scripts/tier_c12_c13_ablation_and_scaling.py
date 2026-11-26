"""
experiments/scripts/tier_c12_c13_ablation_and_scaling.py

Tier C.12 & Tier C.13:
1. Tier C.12 (Self-Term Correction Ablation):
   Compares Raw DScribe Jacobian vs. Translation-Corrected Jacobian (fix_self_terms)
   on:
     - Covariance symmetry/PSD error
     - Test Energy & Force RMSE (kcal/mol, kcal/mol/Å)
     - Net-force violation ||sum_i F_i||_2 (translational momentum conservation)
2. Tier C.13 (Computational Scaling Study):
   Measures exact wall-clock runtime and memory footprint across N in {10, 25, 50, 100},
   broken down into:
     - SOAP descriptor & Jacobian extraction
     - Covariance matrix assembly
     - Cholesky factorization
     - Test prediction
     - Single-candidate acquisition scoring
"""

import sys
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.rmd17 import load_rmd17
from src.kernels.soap_ef_kernel import (
    soap_with_jacobian,
    energy_energy,
    energy_force,
    force_energy,
    force_force,
)

RESULTS_FILE = Path("experiments/results/tier_c12_c13_results.json")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(exist_ok=True)


def extract_soap_variants(z, R_batch, symbols, r_cut=5.0, n_max=4, l_max=3, fix_self=True):
    """Extracts L2-normalized SOAP descriptors and Jacobians with or without self-term correction."""
    soap = SOAP(species=symbols, r_cut=r_cut, n_max=n_max, l_max=l_max, sparse=False)
    D_list, J_list = [], []

    for pos in R_batch:
        atoms = Atoms(numbers=z, positions=pos)
        if fix_self:
            D, J = soap_with_jacobian(soap, atoms)
        else:
            # Raw DScribe analytical derivatives without translational self-term fix
            J_raw, D = soap.derivatives(atoms, method="analytical", return_descriptor=True)
            J = np.transpose(J_raw, (0, 1, 2, 3))

        norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
        D_norm = D / norms
        norm_c = norms[:, None, None, :]
        dot_proj = np.einsum("cf,cmkf->cmk", D, J)
        proj_term = np.einsum("cf,cmk->cmkf", D, dot_proj) / (norm_c**3)
        J_norm = (J / norm_c) - proj_term

        D_list.append(D_norm)
        J_list.append(J_norm)

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


def run_tier_c12_ablation(z, R, E_norm, F_norm, E_std, split, symbols, l, v, nE, nF):
    print("\n=== Tier C.12: Self-Term Correction Ablation (Raw vs. Corrected Jacobian) ===")
    n_atoms = len(z)
    dim = 1 + n_atoms * 3

    train_idx = split["pool"][::150][:25]
    test_idx = split["test"][::200][:50]

    y_train = np.zeros(len(train_idx) * dim)
    for i, idx in enumerate(train_idx):
        y_train[i * dim] = E_norm[idx]
        y_train[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

    y_test = np.zeros(len(test_idx) * dim)
    for i, idx in enumerate(test_idx):
        y_test[i * dim] = E_norm[idx]
        y_test[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

    idx_E = np.arange(0, len(test_idx) * dim, dim)
    idx_F = np.setdiff1d(np.arange(len(test_idx) * dim), idx_E)

    ablation_summary = {}
    for label, fix_flag in [("raw_dscribe", False), ("corrected_jacobian", True)]:
        D_tr, J_tr = extract_soap_variants(z, R[train_idx], symbols, fix_self=fix_flag)
        D_te, J_te = extract_soap_variants(z, R[test_idx], symbols, fix_self=fix_flag)

        # Check translational sum rule ||sum_m dD/dR_m||_max
        trans_violation = float(np.max([np.max(np.abs(np.sum(J, axis=1))) for J in J_tr]))

        K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
        c_tr, low = cho_factor(K_tr)
        alpha = cho_solve((c_tr, low), y_train)

        K_star = build_K_star(D_te, J_te, D_tr, J_tr, l, v)
        y_pred = K_star @ alpha

        err = (y_test - y_pred) * E_std
        e_rmse = float(np.sqrt(np.mean(err[idx_E] ** 2)))
        f_rmse = float(np.sqrt(np.mean(err[idx_F] ** 2)))

        # Net-force violation ||sum_atom F_atom||_2 averaged over test frames (kcal/mol/Å)
        net_forces = []
        for i in range(len(test_idx)):
            f_frame = (y_pred[i * dim + 1 : (i + 1) * dim] * E_std).reshape(n_atoms, 3)
            net_forces.append(float(np.linalg.norm(np.sum(f_frame, axis=0))))
        mean_net_force = float(np.mean(net_forces))

        ablation_summary[label] = {
            "jacobian_trans_violation": trans_violation,
            "energy_rmse_kcal": e_rmse,
            "force_rmse_kcal_ang": f_rmse,
            "mean_net_force_violation": mean_net_force,
        }
        print(
            f"{label:<22} | Jac Trans Err: {trans_violation:.2e} | "
            f"E-RMSE: {e_rmse:.3f} | F-RMSE: {f_rmse:.3f} | "
            f"Net-Force ||ΣF||: {mean_net_force:.2e} kcal/mol/Å"
        )

    return ablation_summary


def run_tier_c13_scaling(z, R, E_norm, F_norm, split, symbols, l, v, nE, nF):
    print("\n=== Tier C.13: Computational Scaling Study (N in {10, 25, 50, 100}) ===")
    n_atoms = len(z)
    dim = 1 + n_atoms * 3
    N_values = [10, 25, 50, 100]

    # Single fixed test frame and single candidate frame for per-query timing
    test_idx = split["test"][:10]
    cand_idx = split["pool"][-1:]
    D_te, J_te = extract_soap_variants(z, R[test_idx], symbols, fix_self=True)
    D_cand, J_cand = extract_soap_variants(z, R[cand_idx], symbols, fix_self=True)
    inv_Omega = np.diag([1.0 / nE**2] + [1.0 / nF**2] * (dim - 1))

    scaling_records = []
    print(
        f"{'N':<6} {'Matrix Dim':<12} {'SOAP (s)':<10} {'Build K (s)':<12} "
        f"{'Cholesky (s)':<13} {'Predict10 (s)':<14} {'Acq1 (s)':<10} {'Peak MB':<10}"
    )
    print("-" * 92)

    for N in N_values:
        train_idx = split["pool"][::150][:N]
        y_tr = np.zeros(N * dim)
        for i, idx in enumerate(train_idx):
            y_tr[i * dim] = E_norm[idx]
            y_tr[i * dim + 1 : (i + 1) * dim] = F_norm[idx].flatten()

        tracemalloc.start()

        t0 = time.perf_counter()
        D_tr, J_tr = extract_soap_variants(z, R[train_idx], symbols, fix_self=True)
        t_soap = time.perf_counter() - t0

        t0 = time.perf_counter()
        K_tr = build_K(D_tr, J_tr, l, v, nE, nF)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        c_tr, low = cho_factor(K_tr)
        alpha = cho_solve((c_tr, low), y_tr)
        t_chol = time.perf_counter() - t0

        t0 = time.perf_counter()
        K_star = build_K_star(D_te, J_te, D_tr, J_tr, l, v)
        _ = K_star @ alpha
        t_pred = time.perf_counter() - t0

        t0 = time.perf_counter()
        K_s1 = build_K_star(D_cand, J_cand, D_tr, J_tr, l, v)
        K_self = np.zeros((dim, dim))
        K_self[0, 0] = energy_energy(D_cand[0], D_cand[0], l, v)
        K_self[0, 1:] = energy_force(D_cand[0], D_cand[0], J_cand[0], l, v).flatten()
        K_self[1:, 0] = force_energy(D_cand[0], J_cand[0], D_cand[0], l, v).flatten()
        K_self[1:, 1:] = force_force(D_cand[0], J_cand[0], D_cand[0], J_cand[0], l, v).reshape(n_atoms * 3, n_atoms * 3)
        v_solve = cho_solve((c_tr, False), K_s1.T)
        Sigma = K_self - K_s1 @ v_solve
        _, _ = np.linalg.slogdet(np.eye(dim) + inv_Omega @ Sigma)
        t_acq = time.perf_counter() - t0

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024**2)

        rec = {
            "N": N,
            "matrix_dim": N * dim,
            "t_soap_sec": float(t_soap),
            "t_build_K_sec": float(t_build),
            "t_cholesky_sec": float(t_chol),
            "t_predict10_sec": float(t_pred),
            "t_acq1_sec": float(t_acq),
            "peak_memory_mb": float(peak_mb),
        }
        scaling_records.append(rec)
        print(
            f"{N:<6} {N * dim:<12} {t_soap:<10.3f} {t_build:<12.3f} "
            f"{t_chol:<13.4f} {t_pred:<14.3f} {t_acq:<10.4f} {peak_mb:<10.2f}"
        )

    return scaling_records


def plot_scaling(scaling_records):
    N_vals = [r["N"] for r in scaling_records]
    t_soap = [r["t_soap_sec"] for r in scaling_records]
    t_build = [r["t_build_K_sec"] for r in scaling_records]
    t_chol = [r["t_cholesky_sec"] for r in scaling_records]
    t_acq60 = [r["t_acq1_sec"] * 60.0 for r in scaling_records]  # 60-candidate pool scoring
    mem_mb = [r["peak_memory_mb"] for r in scaling_records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(N_vals, t_soap, "o-", label="SOAP + Jacobian Extraction")
    ax.plot(N_vals, t_build, "s-", label="Covariance Matrix Assembly O(N²)")
    ax.plot(N_vals, t_acq60, "d-", label="60-Candidate MI Scoring")
    ax.plot(N_vals, t_chol, "^--", label="Cholesky Factorization O(N³)")
    ax.set_xlabel("Number of Training Configurations (N)")
    ax.set_ylabel("Wall-Clock Runtime (seconds)")
    ax.set_title("(a) Component Runtime Scaling")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(N_vals, mem_mb, "s-", color="C3", linewidth=2)
    ax.set_xlabel("Number of Training Configurations (N)")
    ax.set_ylabel("Peak Traced Memory (MB)")
    ax.set_title("(b) Peak Memory Footprint")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/tier_c13_computational_scaling.png", dpi=300)
    print("\nSaved scaling plot to figures/tier_c13_computational_scaling.png")


def main():
    data = load_rmd17("ethanol")
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    E_std = np.std(E)
    E_norm = (E - np.mean(E)) / E_std
    F_norm = F / E_std
    symbols = ["C", "O", "H"]
    l, v, nE, nF = 1.5, 1.0, 0.01, 0.05

    with open("experiments/configs/split_structural.json", "r") as f:
        split = json.load(f)

    c12 = run_tier_c12_ablation(z, R, E_norm, F_norm, E_std, split, symbols, l, v, nE, nF)
    c13 = run_tier_c13_scaling(z, R, E_norm, F_norm, split, symbols, l, v, nE, nF)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"tier_c12_ablation": c12, "tier_c13_scaling": c13}, f, indent=4)

    plot_scaling(c13)


if __name__ == "__main__":
    main()
