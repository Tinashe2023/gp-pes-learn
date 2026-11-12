"""
experiments/scripts/generate_splits.py

Tier A.3: Generates and freezes data splits for the Ethanol rMD17 dataset.
Creates 3 fixed configurations: Random, Contiguous, and Structural (C-O bond gap).
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.rmd17 import load_rmd17

CONFIG_DIR = Path("experiments/configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("--- Tier A.3: Generating Frozen Data Splits for Ethanol ---")
    
    # Load data to get total frames and coordinates for structural split
    data = load_rmd17("ethanol")
    R = data["R"]
    z = data["z"]
    n_frames = len(R)
    print(f"Loaded {n_frames} frames of ethanol.")
    
    # 1. Random Split (80% pool, 20% test)
    rng = np.random.default_rng(42)
    all_idx = np.arange(n_frames)
    rng.shuffle(all_idx)
    split_idx = int(0.8 * n_frames)
    random_pool = all_idx[:split_idx].tolist()
    random_test = all_idx[split_idx:].tolist()
    
    with open(CONFIG_DIR / "split_random.json", "w") as f:
        json.dump({"pool": random_pool, "test": random_test}, f)
    print(f"Saved Random Split (Pool: {len(random_pool)}, Test: {len(random_test)})")
    
    # 2. Contiguous Trajectory Split (First 80% pool, last 20% test)
    contig_pool = list(range(0, split_idx))
    contig_test = list(range(split_idx, n_frames))
    
    with open(CONFIG_DIR / "split_contiguous.json", "w") as f:
        json.dump({"pool": contig_pool, "test": contig_test}, f)
    print(f"Saved Contiguous Split (Pool: {len(contig_pool)}, Test: {len(contig_test)})")
    
    # 3. Structural C-O Bond Gap Split
    c_idx = np.where(z == 6)[0][0]
    o_idx = np.where(z == 8)[0][0]
    distances = np.linalg.norm(R[:, c_idx, :] - R[:, o_idx, :], axis=1)
    
    gap_min = np.mean(distances) + 0.2 * np.std(distances)
    gap_max = np.mean(distances) + 1.2 * np.std(distances)
    
    in_gap = (distances >= gap_min) & (distances <= gap_max)
    struct_test = np.where(in_gap)[0].tolist()
    struct_pool = np.where(~in_gap)[0].tolist()
    
    with open(CONFIG_DIR / "split_structural.json", "w") as f:
        json.dump({
            "pool": struct_pool, 
            "test": struct_test, 
            "gap_min": float(gap_min), 
            "gap_max": float(gap_max)
        }, f)
    print(f"Saved Structural Split (Pool: {len(struct_pool)}, Test: {len(struct_test)})")
    
    print("\nAll splits frozen and saved to experiments/configs/.")

if __name__ == "__main__":
    main()