"""Run from project root: python tests/test_exact_ef_gp.py
No ASE/DScribe required. Checks linear algebra, signs and label scaling only.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.gp_models.exact_ef_gp import ExactEnergyForceGP, covariance_block
from src.kernels.soap_ef_kernel import force_force


def feature(x):
    # Analytic descriptor D=x, one atom; only its x coordinate contributes.
    D = np.array([[x]],dtype=float)
    J = np.zeros((1,1,3,1)); J[0,0,0,0] = 1
    return D,J


def main():
    rng = np.random.default_rng(17)
    a = (rng.normal(size=(2,5)),rng.normal(size=(2,2,3,5)))
    b = (rng.normal(size=(3,5)),rng.normal(size=(3,3,3,5)))
    block = covariance_block(a,b,1.7,2.3)
    ref = force_force(a[0],a[1],b[0],b[1],1.7,2.3).reshape(6,9)
    np.testing.assert_allclose(block[1:,1:],ref,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(block,covariance_block(b,a,1.7,2.3).T,atol=1e-12)
    x = np.linspace(-2,2,15)
    E = 1000 + np.sin(x)
    F = np.zeros((len(x),1,3)); F[:,0,0] = -np.cos(x)
    gp = ExactEnergyForceGP(lengthscale=1.2,energy_noise=1e-5,force_noise=1e-5)
    gp.fit([feature(t) for t in x], E, F)
    z = np.linspace(-1.9,1.9,51)
    ep,fp = gp.predict([feature(t) for t in z])
    er = np.sqrt(np.mean((ep-1000-np.sin(z))**2))
    fr = np.sqrt(np.mean((fp[:,0,0]+np.cos(z))**2))
    assert er < 1e-3 and fr < 1e-3
    np.testing.assert_allclose(fp[:,0,1:],0,atol=1e-12)
    h = 1e-5
    eplus,_ = gp.predict([feature(0.37+h)])
    eminus,_ = gp.predict([feature(0.37-h)])
    _,fmid = gp.predict([feature(0.37)])
    np.testing.assert_allclose(-(eplus-eminus)/(2*h),fmid[:,0,0],atol=2e-6)
    K = np.block([[covariance_block(feature(i),feature(j),1.2,1) for j in x[:4]] for i in x[:4]])
    assert np.linalg.eigvalsh(K).min() > -1e-10
    print('PASS: optimized force block, transpose symmetry, positive semidefiniteness')
    print('PASS: posterior force equals negative energy gradient; zero unused components')
    print(f'Analytic held-out RMSE: energy={er:.3e}, force={fr:.3e}')
    print(f'Log marginal likelihood (normalized labels): {gp.log_marginal_likelihood:.6f}')


if __name__ == '__main__':
    main()
