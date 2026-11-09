"""Small exact energy-force GP. Inputs are precomputed SOAP descriptors/Jacobians.

Per-frame ordering: energy, then forces flattened in atom-major xyz order.
Noise parameters are standard deviations in the original label units.
A single energy scale normalizes BOTH energies and forces, preserving F=-grad E.
Fixed hyperparameters first; marginal likelihood is exposed for later optimization.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve
from src.kernels.soap_ef_kernel import energy_energy, energy_force, force_energy


def covariance_block(a, b, lengthscale, variance):
    """Contract the RBF Hessian without constructing a feature-by-feature tensor."""
    Da, Ja = a
    Db, Jb = b
    u = Da[:, None, :] - Db[None, :, :]
    w = variance * np.exp(-np.sum(u*u, axis=-1)/(2*lengthscale**2))
    A = Ja.reshape(Da.shape[0], -1, Da.shape[1])
    B = Jb.reshape(Db.shape[0], -1, Db.shape[1])
    dot = np.einsum('pq,pif,qjf->ij', w, A, B, optimize=True)
    au = np.einsum('pqf,pif->pqi', u, A, optimize=True)
    bu = np.einsum('pqf,qjf->pqj', u, B, optimize=True)
    ff = dot/lengthscale**2 - np.einsum('pq,pqi,pqj->ij', w, au, bu, optimize=True)/lengthscale**4
    out = np.empty((1+A.shape[1], 1+B.shape[1]))
    out[0,0] = energy_energy(Da, Db, lengthscale, variance)
    out[0,1:] = energy_force(Da, Db, Jb, lengthscale, variance).ravel()
    out[1:,0] = force_energy(Da, Ja, Db, lengthscale, variance).ravel()
    out[1:,1:] = ff
    return out


class ExactEnergyForceGP:
    def __init__(self, lengthscale=3.0, variance=1.0,
                 energy_noise=1e-4, force_noise=1e-3, jitter=1e-10):
        values = [lengthscale, variance, energy_noise, force_noise, jitter]
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError('Hyperparameters must be finite and positive.')
        self.lengthscale, self.variance = lengthscale, variance
        self.energy_noise, self.force_noise, self.jitter = energy_noise, force_noise, jitter

    def fit(self, features, energies, forces):
        self.features = list(features)
        E, F = np.asarray(energies, dtype=float), np.asarray(forces, dtype=float)
        n = len(self.features)
        if n == 0 or E.shape != (n,) or F.ndim != 3 or F.shape[0] != n or F.shape[2] != 3:
            raise ValueError('Expected energies (N,) and forces (N, atoms, 3).')
        if not np.all(np.isfinite(E)) or not np.all(np.isfinite(F)):
            raise ValueError('Labels must be finite.')
        self.width = 1 + F.shape[1]*3
        for D, J in self.features:
            if D.ndim != 2 or J.shape != (D.shape[0], F.shape[1], 3, D.shape[1]):
                raise ValueError('Inconsistent descriptor/Jacobian shape.')
            if not np.all(np.isfinite(D)) or not np.all(np.isfinite(J)):
                raise ValueError('Features must be finite.')
        self.energy_mean = E.mean()
        self.scale = E.std() if E.std() > 1e-12 else 1.0
        y = np.column_stack(((E-self.energy_mean)/self.scale,
                             F.reshape(n,-1)/self.scale)).ravel()
        K = np.empty((n*self.width, n*self.width))
        for i in range(n):
            si = slice(i*self.width,(i+1)*self.width)
            for j in range(i+1):
                sj = slice(j*self.width,(j+1)*self.width)
                block = covariance_block(self.features[i],self.features[j],self.lengthscale,self.variance)
                K[si,sj], K[sj,si] = block, block.T
        noise = np.full(self.width, (self.force_noise/self.scale)**2)
        noise[0] = (self.energy_noise/self.scale)**2
        K.flat[::K.shape[0]+1] += np.tile(noise,n) + self.jitter
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L,True),y)
        self.log_marginal_likelihood = float(-0.5*y@self.alpha - np.log(np.diag(self.L)).sum()
                                           - 0.5*len(y)*np.log(2*np.pi))
        return self

    def predict(self, features):
        """Posterior means in original units; one query frame at a time."""
        if not hasattr(self,'alpha'):
            raise RuntimeError('Call fit before predict.')
        energies, forces = [], []
        for frame in features:
            cross = np.hstack([covariance_block(frame,b,self.lengthscale,self.variance)
                               for b in self.features])
            pred = (cross@self.alpha)*self.scale
            energies.append(pred[0]+self.energy_mean)
            forces.append(pred[1:].reshape(-1,3))
        return np.asarray(energies), np.asarray(forces)
