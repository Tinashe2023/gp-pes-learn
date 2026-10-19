"""
tests/test_soap_ef_kernel.py

Verifies the chain-rule energy/force covariance blocks against brute-force
finite differences of the TRUE pipeline (coordinates -> SOAP -> atom-summed
RBF kernel), recomputing SOAP at every perturbed geometry.

Run from the project root:
    python tests/test_soap_ef_kernel.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from src.kernels.soap_ef_kernel import (
    soap_with_jacobian, energy_energy, energy_force, force_energy, force_force,
)

rng = np.random.default_rng(0)
symbols = ["O", "H", "H"]
base = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
Ra = base + 0.05 * rng.standard_normal((3, 3))
Rb = base + 0.05 * rng.standard_normal((3, 3))

soap = SOAP(species=["O", "H"], r_cut=5.0, n_max=3, l_max=2, sparse=False)
L, V = 3.0, 1.0


def desc(R):
    return soap.create(Atoms(symbols=symbols, positions=R))


def KEE_true(Ra_, Rb_):
    return energy_energy(desc(Ra_), desc(Rb_), L, V)


Da, Ja = soap_with_jacobian(soap, Atoms(symbols=symbols, positions=Ra))
Db, Jb = soap_with_jacobian(soap, Atoms(symbols=symbols, positions=Rb))

cov_EF = energy_force(Da, Db, Jb, L, V)
cov_FE = force_energy(Da, Ja, Db, L, V)
cov_FF = force_force(Da, Ja, Db, Jb, L, V)

h = 1e-4


def shifted(R, m, k, s):
    R2 = R.copy(); R2[m, k] += s
    return R2


print("--- Energy-Force (b side): ALL 9 entries ---")
worst_ef = 0.0
for m in range(3):
    for k in range(3):
        num = -(KEE_true(Ra, shifted(Rb, m, k, h)) - KEE_true(Ra, shifted(Rb, m, k, -h))) / (2 * h)
        err = abs(num - cov_EF[m, k]) / (abs(num) + 1e-9)
        worst_ef = max(worst_ef, err)

print("--- Force-Energy (a side): ALL 9 entries ---")
worst_fe = 0.0
for m in range(3):
    for k in range(3):
        num = -(KEE_true(shifted(Ra, m, k, h), Rb) - KEE_true(shifted(Ra, m, k, -h), Rb)) / (2 * h)
        err = abs(num - cov_FE[m, k]) / (abs(num) + 1e-9)
        worst_fe = max(worst_fe, err)

print("--- Force-Force: ALL 81 entries ---")
worst_ff = 0.0
for m in range(3):
    for k in range(3):
        for n in range(3):
            for l in range(3):
                f_pp = KEE_true(shifted(Ra, m, k, h),  shifted(Rb, n, l, h))
                f_pm = KEE_true(shifted(Ra, m, k, h),  shifted(Rb, n, l, -h))
                f_mp = KEE_true(shifted(Ra, m, k, -h), shifted(Rb, n, l, h))
                f_mm = KEE_true(shifted(Ra, m, k, -h), shifted(Rb, n, l, -h))
                num = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                err = abs(num - cov_FF[m, k, n, l]) / (abs(num) + 1e-6)
                worst_ff = max(worst_ff, err)

print(f"\nWorst relative error, energy-force: {worst_ef:.2e}")
print(f"Worst relative error, force-energy: {worst_fe:.2e}")
print(f"Worst relative error, force-force:  {worst_ff:.2e}")

tol = 1e-4
assert worst_ef < tol and worst_fe < tol and worst_ff < tol, "Covariance blocks disagree with finite differences"
print("\nPASSED: all covariance blocks match finite differences of the true pipeline.")
