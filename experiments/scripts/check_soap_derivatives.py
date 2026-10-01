"""
experiments/scripts/check_soap_derivatives.py

Before building anything on top of dscribe's SOAP derivatives, verify
they're actually correct on our data. There's a known, still-open bug
(dscribe GitHub issue #113) where the 'analytical' derivative method
disagrees with the numerical (finite-difference) one whenever a smooth
weighting function is used in SOAP - and since our whole approach
depends on force/derivative observations being correct, this needs to
be checked, not assumed.

Deliberate choice: we do NOT set a `weighting` argument below (SOAP's
default is a hard cutoff, no smoothing function), specifically to stay
on the side of this bug where analytical derivatives are reported to
be correct. This script confirms that choice actually holds here.

Run from the project root:
    python experiments/scripts/check_soap_derivatives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from src.utils.rmd17 import load_rmd17

data = load_rmd17("ethanol")
z, R = data["z"], data["R"]

atoms = Atoms(numbers=z, positions=R[0])
species = sorted(set(atoms.get_chemical_symbols()))
print(f"Species in system: {species}")

# Deliberately no `weighting` argument - see module docstring.
soap = SOAP(species=species, r_cut=5.0, n_max=4, l_max=3, sparse=False)

print("\nComputing analytical derivatives...")
der_analytical, des_analytical = soap.derivatives(
    atoms, method="analytical", return_descriptor=True
)

print("Computing numerical derivatives (slower, finite-difference)...")
der_numerical, des_numerical = soap.derivatives(
    atoms, method="numerical", return_descriptor=True
)

print(f"\nDescriptor shape:            {des_analytical.shape}")
print(f"Analytical derivative shape: {der_analytical.shape}")
print(f"Numerical derivative shape:  {der_numerical.shape}")

desc_diff = np.abs(des_analytical - des_numerical).max()
der_diff = np.abs(der_analytical - der_numerical)
der_scale = np.abs(der_numerical).max()

print("\n--- Agreement check ---")
print(f"Max |descriptor difference|:            {desc_diff:.3e}")
print(f"Max |derivative difference|:             {der_diff.max():.3e}")
print(f"Max |numerical derivative| (for scale):  {der_scale:.3e}")
print(f"Max relative derivative error:           {der_diff.max() / der_scale:.3e}")

tol = 1e-4
if der_diff.max() / der_scale < tol:
    print(f"\nPASSED: analytical and numerical derivatives agree "
          f"(relative error < {tol}). Safe to use method='analytical' "
          f"(fast) going forward with this SOAP configuration.")
else:
    print(f"\nFAILED: derivatives disagree by more than expected. "
          f"Do NOT trust 'analytical' here - fall back to "
          f"method='numerical' (slower but verified) until this is "
          f"root-caused.")