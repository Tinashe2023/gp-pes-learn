"""Re-render manuscript figures 4–7 from saved benchmark results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load_result(name):
    with (RESULTS / name).open(encoding="utf-8") as result_file:
        return json.load(result_file)


def plot_b6():
    results = load_result("tier_b6_b7_multiseed.json")
    n_init, n_steps = 5, 16
    steps = np.arange(n_init, n_init + n_steps)
    styles = {
        "random": ("gray", "o", "--", "Random Selection"),
        "max_variance": ("C3", "s", "-", "Max-Variance (Trace)"),
        "qbc": ("C2", "^", "-.", "Committee Disagreement (QBC)"),
        "d_optimality": ("C1", "v", ":", "D-Optimality (Unweighted)"),
        "info_gain": ("C0", "d", "-", "Info-Gain (Log-Det MI)"),
    }
    metrics = [
        ("force_rmse", "Gap Force RMSE (kcal/mol/Å)", "Gap Force RMSE (kcal/mol/Å)"),
        ("energy_rmse", "Gap Energy RMSE (kcal/mol)", "Gap Energy RMSE (kcal/mol)"),
        ("max_pred_var", "Max Pool Predictive Trace Variance", "Max Pool Predictive Trace Variance"),
        ("cum_info_gain", "Cumulative Information Gain", "Cumulative Information Gain (nats)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for (key, title, ylabel), ax in zip(metrics, axes.flat):
        for strategy, (color, marker, linestyle, label) in styles.items():
            runs = results.get(strategy, [])
            if not runs:
                continue
            values = np.asarray([run[key] for run in runs])
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(steps, mean, marker=marker, linestyle=linestyle, color=color,
                    label=label, markersize=5)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(f"{['(a)', '(b)', '(c)', '(d)'][list(axes.flat).index(ax)]} {title}", fontsize=10.5)
        ax.set_xlabel("Training Set Size (N)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        if key == "force_rmse":
            ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "tier_b6_b7_learning_curves.png", dpi=300)
    plt.close(fig)


def plot_b8():
    results = load_result("tier_b8_batch_results.json")
    batch_sizes = [1, 5, 10, 25]
    colors = {1: "C0", 5: "C2", 10: "C1", 25: "C3"}
    markers = {1: "d", 5: "s", 10: "^", 25: "o"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for batch in batch_sizes:
        runs = results[str(batch)]
        n_vals = runs[0]["N_history"]
        values = np.asarray([run["force_rmse"] for run in runs])
        mean, std = values.mean(axis=0), values.std(axis=0)
        axes[0].plot(n_vals, mean, marker=markers[batch], color=colors[batch],
                     label=f"Batch B = {batch}")
        axes[0].fill_between(n_vals, mean - std, mean + std, color=colors[batch], alpha=0.15)
    axes[0].set(xlabel="Training Set Size (N)", ylabel="Gap Force RMSE (kcal/mol/Å)",
                title="(a) Accuracy vs. Query Budget")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()
    mean_times = [np.mean([r["wall_time_sec"] for r in results[str(b)]]) for b in batch_sizes]
    std_times = [np.std([r["wall_time_sec"] for r in results[str(b)]]) for b in batch_sizes]
    axes[1].bar([f"B={b}" for b in batch_sizes], mean_times, yerr=std_times, capsize=4,
                color=[colors[b] for b in batch_sizes], alpha=0.85)
    axes[1].set(ylabel="Total Wall-Clock Time (s)", title="(b) Retraining & Acquisition Runtime")
    axes[1].grid(True, axis="y", alpha=0.4)
    multi_batch = [b for b in batch_sizes if b > 1]
    mean_div = [np.mean([r["mean_batch_diversity"] for r in results[str(b)]]) for b in multi_batch]
    std_div = [np.std([r["mean_batch_diversity"] for r in results[str(b)]]) for b in multi_batch]
    axes[2].bar([f"B={b}" for b in multi_batch], mean_div, yerr=std_div, capsize=4,
                color=[colors[b] for b in multi_batch], alpha=0.85)
    axes[2].set(ylabel="Mean Pairwise SOAP Distance", title="(c) Intra-Batch Structural Diversity")
    axes[2].grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "tier_b8_batch_analysis.png", dpi=300)
    plt.close(fig)


def plot_b9():
    results = load_result("tier_b9_d15_validation.json")
    predicted = results["predicted_ig"]
    realized = results["realized_pool_mi_drop"]
    gamma = results["cum_gamma_T"]
    steps = np.arange(1, len(predicted) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(predicted, realized, c=steps, cmap="viridis", edgecolor="k", s=55)
    slope, intercept = np.polyfit(predicted, realized, 1)
    x_line = np.linspace(min(predicted), max(predicted), 50)
    axes[0].plot(x_line, slope*x_line + intercept, "r--", linewidth=1.5,
                 label=f"Linear Fit (r = {results['pearson_r_mi']:.2f})")
    axes[0].set(xlabel=r"Predicted Acquisition Score $I(\mathbf{y}_*; f \mid \mathcal{D}_t)$ [nats]",
                ylabel="Realized Global Pool Entropy Reduction [nats]",
                title="(a) Acquisition Score Validation")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()
    axes[1].plot(steps, gamma, "d-", color="C0", label=r"Empirical $\gamma_T$ (Log-Det MI)")
    envelope = gamma[-1] / np.log(1 + len(steps)) * np.log(1 + steps)
    axes[1].plot(steps, envelope, "k--", label=r"Sublinear Envelope $\mathcal{O}(d_{\mathrm{eff}} \log T)$")
    axes[1].set(xlabel="Active Learning Step (T)",
                ylabel=r"Cumulative Information Gain $\gamma_T$ [nats]",
                title="(b) Sublinear Information Gain Growth")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()
    axes[2].plot(steps, results["max_var_T"], "s-", color="C3", label="Max Pool Trace Variance")
    axes[2].plot(steps, results["p95_var_T"], "^--", color="C1", label="95th Percentile Variance")
    axes[2].plot(steps, results["mean_var_T"], "o-", color="C2", label="Mean Pool Variance")
    theory = results["mean_var_T"][0] * (np.asarray(gamma) / steps) / gamma[0]
    axes[2].plot(steps, theory, "k:", linewidth=1.8,
                 label=r"Theoretical Envelope $\mathcal{O}(\gamma_T / T)$")
    axes[2].set(xlabel="Active Learning Step (T)", ylabel="Posterior Predictive Trace Variance",
                title="(c) Posterior Variance Decay")
    axes[2].grid(True, alpha=0.4)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "tier_b9_d15_d16_validation.png", dpi=300)
    plt.close(fig)


def plot_c14():
    results = load_result("tier_c14_robustness.json")
    steps = np.arange(5, 21)
    panels = [
        ("ethanol_random", "(a) Ethanol: Random Split (Interpolation)"),
        ("ethanol_contiguous", "(b) Ethanol: Contiguous Trajectory Split"),
        ("ethanol_structural", "(c) Ethanol: Structural C–O Gap Split"),
        ("malonaldehyde_structural", "(d) Malonaldehyde: Structural O–H Gap Split"),
    ]
    styles = {
        "random": ("gray", "o", "--", "Random Sampling"),
        "max_variance": ("C3", "s", "-", "Max-Variance (Trace)"),
        "info_gain": ("C0", "d", "-", "Info-Gain (Log-Det MI)"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for (key, title), ax in zip(panels, axes.flat):
        for strategy, (color, marker, linestyle, label) in styles.items():
            values = np.asarray([run["force_rmse"] for run in results[key][strategy]])
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(steps, mean, marker=marker, linestyle=linestyle, color=color,
                    label=label, markersize=4.5)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Training Set Size (N)")
        ax.set_ylabel("Force RMSE (kcal/mol/Å)")
        ax.grid(True, alpha=0.4)
        if key == "ethanol_random":
            ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "tier_c14_multi_split_and_molecule.png", dpi=300)
    plt.close(fig)


def plot_c13():
    results = load_result("tier_c12_c13_results.json")["tier_c13_scaling"]
    n_vals = [record["N"] for record in results]
    t_soap = [record["t_soap_sec"] for record in results]
    t_build = [record["t_build_K_sec"] for record in results]
    t_cholesky = [record["t_cholesky_sec"] for record in results]
    t_acq60 = [record["t_acq1_sec"] * 60 for record in results]
    memory = [record["peak_memory_mb"] for record in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(n_vals, t_soap, "o-", label="SOAP + Jacobian Extraction")
    axes[0].plot(n_vals, t_build, "s-", label="Covariance Matrix Assembly O(N²)")
    axes[0].plot(n_vals, t_acq60, "d-", label="60-Candidate MI Scoring")
    axes[0].plot(n_vals, t_cholesky, "^--", label="Cholesky Factorization O(N³)")
    axes[0].set(xlabel="Number of Training Configurations (N)",
                ylabel="Wall-Clock Runtime (seconds)", title="(a) Component Runtime Scaling")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(fontsize=8.5)
    axes[1].plot(n_vals, memory, "s-", color="C3", linewidth=2)
    axes[1].set(xlabel="Number of Training Configurations (N)",
                ylabel="Peak Traced Memory (MB)", title="(b) Peak Memory Footprint")
    axes[1].grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "tier_c13_computational_scaling.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_b6()
    plot_b8()
    plot_b9()
    plot_c14()
    plot_c13()
    print("Regenerated clean figures 4–7 and Tier C.13 from saved JSON results.")
