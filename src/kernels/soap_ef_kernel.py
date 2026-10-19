"""
src/kernels/soap_ef_kernel.py

Atom-summed RBF kernel on SOAP descriptors, with the covariance blocks
needed for joint energy + force Gaussian-process regression.

Model: total energy E(R) = sum_i eps(D_i(R)), eps ~ GP(0, k_RBF) on the
SOAP descriptor D_i of atom i. Forces are F = -dE/dR. Covariances follow
from the chain rule through the SOAP Jacobian dD_i/dR_{m,k}.

Index conventions (matching dscribe's derivative array):
    D   : (n_atoms, d)                 descriptors, one row per center atom
    J   : (n_atoms, n_atoms, 3, d)     J[c, m, k, :] = dD_c / dR_{m,k}

IMPORTANT: dscribe returns ZEROS for the self-terms J[c, c, :, :] (a
center atom's descriptor differentiated w.r.t. its own position).
`soap_with_jacobian` restores them using translation invariance:
    dD_c/dR_c = - sum_{j != c} dD_c/dR_j
"""

import numpy as np


def fix_self_terms(der):
    der = der.copy()
    n = der.shape[0]
    for c in range(n):
        others = [j for j in range(n) if j != c]
        der[c, c] = -der[c, others].sum(axis=0)
    return der


def soap_with_jacobian(soap, atoms):
    """Descriptors and a complete (self-terms restored) Jacobian."""
    der, des = soap.derivatives(atoms, method="analytical", return_descriptor=True)
    return des, fix_self_terms(der)


def _rbf(D1, D2, lengthscale, variance):
    diff = D1[:, None, :] - D2[None, :, :]                 # (n1, n2, d)
    K = variance * np.exp(-np.sum(diff ** 2, axis=-1) / (2 * lengthscale ** 2))
    return K, diff


def energy_energy(Da, Db, lengthscale, variance):
    """Cov(E^a, E^b) = sum_{p,q} k(D^a_p, D^b_q). Scalar."""
    K, _ = _rbf(Da, Db, lengthscale, variance)
    return K.sum()


def energy_force(Da, Db, Jb, lengthscale, variance):
    """Cov(E^a, F^b_{m,k}) = -dK_EE/dR^b_{m,k}.  Shape (n_b, 3)."""
    K, diff = _rbf(Da, Db, lengthscale, variance)
    grad_D2 = K[:, :, None] * diff / lengthscale ** 2     # dk/dD2
    return -np.einsum("pqf,qmkf->mk", grad_D2, Jb)


def force_energy(Da, Ja, Db, lengthscale, variance):
    """Cov(F^a_{m,k}, E^b) = -dK_EE/dR^a_{m,k}.  Shape (n_a, 3)."""
    K, diff = _rbf(Da, Db, lengthscale, variance)
    grad_D1 = -K[:, :, None] * diff / lengthscale ** 2    # dk/dD1
    return -np.einsum("pqf,pmkf->mk", grad_D1, Ja)


def force_force(Da, Ja, Db, Jb, lengthscale, variance):
    """Cov(F^a_{m,k}, F^b_{n,l}) = +d2K_EE/dR^a_{m,k} dR^b_{n,l}.
    Shape (n_a, 3, n_b, 3)."""
    K, diff = _rbf(Da, Db, lengthscale, variance)
    d = Da.shape[1]
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    hess = K[:, :, None, None] * (np.eye(d)[None, None] / lengthscale ** 2
                                   - outer / lengthscale ** 4)   # d2k/dD1dD2
    return np.einsum("pqfg,pmkf,qnlg->mknl", hess, Ja, Jb)
