"""
experiments/scripts/load_ethanol.py

First real-data check: load ethanol from rMD17, confirm shapes and
units, and run a basic physical sanity check before any modeling
work touches this data.

Run from the project root:
    python experiments/scripts/load_ethanol.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
from ase import Atoms

from src.utils.rmd17 import load_rmd17

data = load_rmd17("ethanol")
z, R, E, F = data["z"], data["R"], data["E"], data["F"]

n_frames, n_atoms, _ = R.shape
first_atoms = Atoms(numbers=z, positions=R[0])

print("--- Shapes ---")
print(f"z (nuclear charges): {z.shape}  dtype={z.dtype}")
print(f"R (coordinates):     {R.shape}  dtype={R.dtype}  (Angstrom)")
print(f"E (energies):         {E.shape}  dtype={E.dtype}  (kcal/mol)")
print(f"F (forces):           {F.shape}  dtype={F.dtype}  (kcal/mol/Angstrom)")
print(f"\nMolecule: {first_atoms.get_chemical_formula()}  "
      f"({n_atoms} atoms, {n_frames} frames)")

print("\n--- Energy statistics (kcal/mol) ---")
print(f"min:  {E.min():.3f}")
print(f"max:  {E.max():.3f}")
print(f"mean: {E.mean():.3f}")
print(f"std:  {E.std():.3f}")
print(
    "Note: these are NOT zero-centered - the absolute scale includes "
    "the molecule's total electronic energy. Model (E - E.mean()), "
    "not raw E, since the GP's constant-mean prior assumes an O(1) "
    "target scale, not a value in the thousands."
)

print("\n--- Force statistics (kcal/mol/Angstrom) ---")
force_magnitudes = np.linalg.norm(F, axis=-1)  # (n_frames, n_atoms)
print(f"mean |F| per atom: {force_magnitudes.mean():.3f}")
print(f"max  |F| per atom: {force_magnitudes.max():.3f}")

# Real physical sanity check, not a formality: for an isolated
# molecule with no external field, the net force summed over all
# atoms in a frame must be ~0 (translational invariance / Newton's
# third law internally). If this fails, something is wrong with how
# the array was parsed (e.g. wrong axis, wrong molecule's forces).
net_force = F.sum(axis=1)                       # (n_frames, 3)
net_force_magnitude = np.linalg.norm(net_force, axis=-1)

print("\n--- Sanity check: net force per frame (should be ~0) ---")
print(f"mean |net force|: {net_force_magnitude.mean():.5f}")
print(f"max  |net force|: {net_force_magnitude.max():.5f}")
assert net_force_magnitude.mean() < 1.0, (
    "Net force is not close to zero relative to individual atomic "
    "forces - check units/parsing before going further."
)
print(f"Passed: net force is small relative to individual atomic "
      f"forces (typical |F| per atom ~ {force_magnitudes.mean():.2f}).")

# Confirm this plugs into the standard atomistic tooling (ASE) we'll
# need for SOAP descriptors in the next step.
print("\n--- First frame as an ASE Atoms object ---")
print(first_atoms)