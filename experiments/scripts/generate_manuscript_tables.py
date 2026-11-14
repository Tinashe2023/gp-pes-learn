"""
experiments/scripts/generate_manuscript_tables.py

Reads all saved JSON outputs from experiments/results/ and writes permanent,
publication-ready LaTeX (.tex) and CSV (.csv) tables into tables/ for direct
inclusion in the AIP manuscript.
"""

import json
import csv
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("experiments/results")
TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def write_csv_and_tex(stem, headers, rows, caption, label):
    csv_path = TABLES_DIR / f"{stem}.csv"
    tex_path = TABLES_DIR / f"{stem}.tex"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        escaped = [str(cell).replace("±", r"$\pm$").replace("×", r"$\times$") for cell in row]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {tex_path} and {csv_path}")


def build_table1_verification_and_c12():
    with open(RESULTS_DIR / "tier_c12_c13_results.json", "r") as f:
        c12 = json.load(f)["tier_c12_ablation"]

    raw = c12["raw_dscribe"]
    corr = c12["corrected_jacobian"]

    headers = [
        "Jacobian Variant",
        "Trans. Sum Error",
        "Net Force ||ΣF|| (kcal/mol/Å)",
        "Energy RMSE (kcal/mol)",
        "Force RMSE (kcal/mol/Å)",
    ]
    rows = [
        [
            "Raw DScribe (Uncorrected)",
            f"{raw['jacobian_trans_violation']:.2e}",
            f"{raw['mean_net_force_violation']:.2e}",
            f"{raw['energy_rmse_kcal']:.3f}",
            f"{raw['force_rmse_kcal_ang']:.3f}",
        ],
        [
            "Corrected Jacobian (Ours)",
            f"{corr['jacobian_trans_violation']:.2e}",
            f"{corr['mean_net_force_violation']:.2e}",
            f"{corr['energy_rmse_kcal']:.3f}",
            f"{corr['force_rmse_kcal_ang']:.3f}",
        ],
    ]
    write_csv_and_tex(
        "table1_self_term_ablation",
        headers,
        rows,
        "Effect of the analytical SOAP Jacobian self-term correction on translational invariance, net-force conservation, and test accuracy on rMD17 ethanol.",
        "tab:self_term_ablation",
    )


def build_table2_hyperparams():
    # Records from Tier A.5 NLML optimization vs Fixed Baseline
    headers = [
        "Configuration",
        "Final NLML",
        "Lengthscale (l)",
        "Variance (v)",
        "Noise n_E",
        "Noise n_F",
        "Val E-RMSE",
        "Val F-RMSE",
    ]
    rows = [
        ["Fixed Baseline (Frozen)", "—", "1.500", "1.000", "0.0100", "0.0500", "4.980", "29.418"],
        ["NLML Opt. (Init 1)", "1297.46", "1.201", "0.010", "1.0270", "6.5509", "5.126", "29.530"],
        ["NLML Opt. (Init 2)", "1297.40", "0.507", "0.011", "1.0090", "6.5664", "—", "—"],
        ["NLML Opt. (Init 3)", "1297.46", "1.422", "0.010", "1.0241", "6.5695", "—", "—"],
    ]
    write_csv_and_tex(
        "table2_hyperparameter_study",
        headers,
        rows,
        "Comparison of validation-backed fixed hyperparameters against marginal-likelihood optimization across three initializations, illustrating high-noise collapse.",
        "tab:hyperparam_study",
    )


def build_table3_acquisition():
    with open(RESULTS_DIR / "tier_b6_b7_multiseed.json", "r") as f:
        res = json.load(f)

    name_map = {
        "random": "Random Selection",
        "max_variance": "Max-Variance (Trace)",
        "qbc": "Committee Disagreement (QBC)",
        "d_optimality": "D-Optimality (Unweighted)",
        "info_gain": "Info-Gain (Log-Det MI, Ours)",
    }
    headers = [
        "Acquisition Strategy",
        "Gap Force RMSE (kcal/mol/Å)",
        "Gap Energy RMSE (kcal/mol)",
        "Cumulative Info Gain (nats)",
        "Runtime / Run (s)",
    ]
    rows = []
    for key, label in name_map.items():
        runs = res[key]
        f_end = [r["force_rmse"][-1] for r in runs]
        e_end = [r["energy_rmse"][-1] for r in runs]
        ig_end = [r["cum_info_gain"][-1] for r in runs]
        tm = [r["wall_time_sec"] for r in runs]
        rows.append([
            label,
            f"{np.mean(f_end):.3f} ± {np.std(f_end):.3f}",
            f"{np.mean(e_end):.3f} ± {np.std(e_end):.3f}",
            f"{np.mean(ig_end):.2f} ± {np.std(ig_end):.2f}",
            f"{np.mean(tm):.1f}",
        ])

    write_csv_and_tex(
        "table3_acquisition_comparison",
        headers,
        rows,
        "Multi-seed active learning performance (mean ± 1σ across 5 seeds at N=20) on the ethanol C–O structural gap.",
        "tab:acquisition_comparison",
    )


def build_table4_batch():
    with open(RESULTS_DIR / "tier_b8_batch_results.json", "r") as f:
        res = json.load(f)

    headers = [
        "Batch Size (B)",
        "Final Gap F-RMSE (kcal/mol/Å)",
        "Final Gap E-RMSE (kcal/mol)",
        "Intra-Batch SOAP Diversity",
        "Total Wall Time (s)",
    ]
    rows = []
    for B in ["1", "5", "10", "25"]:
        runs = res[B]
        f_end = [r["force_rmse"][-1] for r in runs]
        e_end = [r["energy_rmse"][-1] for r in runs]
        div = [r["mean_batch_diversity"] for r in runs]
        tm = [r["wall_time_sec"] for r in runs]
        div_str = "—" if B == "1" else f"{np.mean(div):.3f} ± {np.std(div):.3f}"
        rows.append([
            f"B = {B}",
            f"{np.mean(f_end):.3f} ± {np.std(f_end):.3f}",
            f"{np.mean(e_end):.3f} ± {np.std(e_end):.3f}",
            div_str,
            f"{np.mean(tm):.1f} ± {np.std(tm):.1f}",
        ])

    write_csv_and_tex(
        "table4_batch_analysis",
        headers,
        rows,
        "Greedy submodular batch active learning performance across batch sizes B ∈ {1, 5, 10, 25} at a fixed budget of N=30 configurations (3 seeds).",
        "tab:batch_analysis",
    )


def build_table5_scaling():
    with open(RESULTS_DIR / "tier_c12_c13_results.json", "r") as f:
        scaling = json.load(f)["tier_c13_scaling"]

    headers = [
        "N",
        "Matrix Dim",
        "SOAP + Jac (s)",
        "Assemble K (s)",
        "Cholesky (s)",
        "Predict 10 (s)",
        "Score 1 Cand (s)",
        "Peak Memory (MB)",
    ]
    rows = []
    for r in scaling:
        rows.append([
            str(r["N"]),
            str(r["matrix_dim"]),
            f"{r['t_soap_sec']:.3f}",
            f"{r['t_build_K_sec']:.3f}",
            f"{r['t_cholesky_sec']:.4f}",
            f"{r['t_predict10_sec']:.3f}",
            f"{r['t_acq1_sec']:.4f}",
            f"{r['peak_memory_mb']:.2f}",
        ])

    write_csv_and_tex(
        "table5_computational_scaling",
        headers,
        rows,
        "Component wall-clock runtime and peak memory scaling on rMD17 ethanol as a function of training set size N.",
        "tab:computational_scaling",
    )


def build_table6_robustness():
    with open(RESULTS_DIR / "tier_c14_robustness.json", "r") as f:
        res = json.load(f)

    regimes = [
        ("ethanol_random", "Ethanol (Random Split)"),
        ("ethanol_contiguous", "Ethanol (Contiguous Split)"),
        ("ethanol_structural", "Ethanol (C–O Gap Split)"),
        ("malonaldehyde_structural", "Malonaldehyde (O–H Gap Split)"),
    ]
    strat_names = {
        "random": "Random",
        "max_variance": "Max-Variance",
        "info_gain": "Info-Gain (Ours)",
    }
    headers = [
        "System & Split Regime",
        "Strategy",
        "Final Force RMSE (kcal/mol/Å)",
        "Final Energy RMSE (kcal/mol)",
    ]
    rows = []
    for r_key, r_label in regimes:
        for s_key, s_label in strat_names.items():
            runs = res[r_key][s_key]
            f_end = [r["force_rmse"][-1] for r in runs]
            e_end = [r["energy_rmse"][-1] for r in runs]
            rows.append([
                r_label,
                s_label,
                f"{np.mean(f_end):.3f} ± {np.std(f_end):.3f}",
                f"{np.mean(e_end):.3f} ± {np.std(e_end):.3f}",
            ])

    write_csv_and_tex(
        "table6_multi_split_robustness",
        headers,
        rows,
        "Cross-split and cross-molecule generalization at N=20 configurations (mean ± 1σ across 3 seeds).",
        "tab:multi_split_robustness",
    )


def main():
    print("--- Generating Permanent Manuscript Tables (.tex and .csv) ---")
    build_table1_verification_and_c12()
    build_table2_hyperparams()
    build_table3_acquisition()
    build_table4_batch()
    build_table5_scaling()
    build_table6_robustness()
    print("\nAll 6 publication tables saved to tables/.")


if __name__ == "__main__":
    main()