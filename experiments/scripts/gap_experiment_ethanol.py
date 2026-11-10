"""
experiments/scripts/gap_experiment_ethanol.py

Step 3: Define a structural coordinate split (C-O bond distance gap)
on rMD17 ethanol to prepare for active learning extrapolation experiments.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.rmd17 import load_rmd17

def main():
    print("--- Step 3: Structuring the Extrapolation Gap Experiment ---")
    data = load_rmd17("ethanol")
    R = data["R"]
    z = data["z"]
    
    # Identify indices for Carbon (6) and Oxygen (8)
    c_indices = np.where(z == 6)[0]
    o_indices = np.where(z == 8)[0]
    
    c_idx = c_indices[0]
    o_idx = o_indices[0]
    
    # Compute C-O bond length across all 100,000 frames
    distances = np.linalg.norm(R[:, c_idx, :] - R[:, o_idx, :], axis=1)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    print(f"C-O Bond Distance (Å): Min={distances.min():.3f}, Max={distances.max():.3f}, Mean={mean_dist:.3f}")
    
    # Define an artificial extrapolation gap around the stretched region
    gap_min = mean_dist + 0.2 * std_dist
    gap_max = mean_dist + 1.2 * std_dist
    
    in_gap = (distances >= gap_min) & (distances <= gap_max)
    out_of_gap = ~in_gap
    
    idx_gap = np.where(in_gap)[0]
    idx_pool = np.where(out_of_gap)[0]
    
    print(f"\nDataset Partitioning:")
    print(f"  Total frames: {len(R)}")
    print(f"  Extrapolation Gap (Test Set) [{gap_min:.3f} Å - {gap_max:.3f} Å]: {len(idx_gap)} frames")
    print(f"  Candidate Pool (Training Candidates): {len(idx_pool)} frames")
    
    # Plot histogram showing the structural gap
    plt.figure(figsize=(8, 4))
    plt.hist(distances[out_of_gap], bins=60, alpha=0.7, label='Candidate Pool (Outside Gap)', color='C0')
    plt.hist(distances[in_gap], bins=30, alpha=0.8, label='Extrapolation Gap (Test Target)', color='C3')
    plt.axvline(gap_min, color='k', linestyle='--', label=f'Gap Limits ({gap_min:.2f}Å - {gap_max:.2f}Å)')
    plt.axvline(gap_max, color='k', linestyle='--')
    plt.xlabel('C–O Bond Length (Å)')
    plt.ylabel('Frame Count')
    plt.title('Ethanol Extrapolation Gap Split (Step 3)')
    plt.legend()
    plt.tight_layout()
    
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/ethanol_gap_split.png", dpi=300)
    print("\nSaved distribution plot to figures/ethanol_gap_split.png")

if __name__ == "__main__":
    main()