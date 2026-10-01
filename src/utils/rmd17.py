"""
src/utils/rmd17.py

Minimal loader for the revised MD17 (rMD17) dataset.

File format (confirmed from the official rMD17 documentation,
Christensen & von Lilienfeld, 2020):
    'nuclear_charges' : nuclear charge of each atom,         shape (n_atoms,)
    'coords'          : Cartesian coordinates, Angstrom,     shape (n_frames, n_atoms, 3)
    'energies'        : total energy, kcal/mol,               shape (n_frames,)
    'forces'          : Cartesian forces, kcal/mol/Angstrom,  shape (n_frames, n_atoms, 3)
    'old_indices', 'old_energies', 'old_forces' : from the original
        (noisier) MD17 dataset - not used here, do not mix with the
        revised 'energies'/'forces' above.

Energies and forces were recomputed at the PBE/def2-SVP level with
tight SCF convergence - a different level of theory than the
original MD17 dataset.
"""

from pathlib import Path
import numpy as np

MOLECULES = (
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
)


def load_rmd17(molecule: str, data_dir="data/raw/rmd17"):
    """
    Load one rMD17 molecule.

    Parameters
    ----------
    molecule : one of MOLECULES, e.g. "ethanol"
    data_dir : path to the extracted rmd17 folder (the folder created
               by `tar -xjf rmd17.tar.bz2`, containing npz_data/)

    Returns
    -------
    dict with keys:
        z : (n_atoms,)              int   nuclear charges
        R : (n_frames, n_atoms, 3)  float Cartesian coordinates, Angstrom
        E : (n_frames,)             float total energy, kcal/mol
        F : (n_frames, n_atoms, 3)  float Cartesian forces, kcal/mol/Angstrom
    """
    if molecule not in MOLECULES:
        raise ValueError(f"Unknown molecule '{molecule}'. Choose from {MOLECULES}")

    data_dir = Path(data_dir)
    candidates = [
        data_dir / "npz_data" / f"rmd17_{molecule}.npz",
        data_dir / f"rmd17_{molecule}.npz",
    ]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(
            f"Could not find rmd17_{molecule}.npz. Checked:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\nPass the correct data_dir if your extraction landed elsewhere."
        )

    data = np.load(npz_path)
    return {
        "z": data["nuclear_charges"],
        "R": data["coords"],
        "E": data["energies"],
        "F": data["forces"],
    }