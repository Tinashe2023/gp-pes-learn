(gp-pes-active-learning) C:\Users\TinasheHando\Desktop\2026 Research\gp-pes-active-learning>python tests/test_soap_ef_kernel.py
--- Energy-Force (b side): ALL 9 entries ---
--- Force-Energy (a side): ALL 9 entries ---
--- Force-Force: ALL 81 entries ---

Worst relative error, energy-force: 4.57e-07
Worst relative error, force-energy: 5.53e-07
Worst relative error, force-force: 5.91e-06

PASSED: all covariance blocks match finite differences of the true pipeline.

(gp-pes-active-learning) C:\Users\TinasheHando\Desktop\2026 Research\gp-pes-active-learning>python tests/test_exact_ef_gp.py
PASS: optimized force block, transpose symmetry, positive semidefiniteness
PASS: posterior force equals negative energy gradient; zero unused components
Analytic held-out RMSE: energy=3.255e-07, force=3.961e-06
Log marginal likelihood (normalized labels): 492.160432

(gp-pes-active-learning) C:\Users\TinasheHando\Desktop\2026 Research\gp-pes-active-learning>python tests/test_soap_ef_kernel.py
--- Energy-Force (b side): ALL 9 entries ---
--- Force-Energy (a side): ALL 9 entries ---
--- Force-Force: ALL 81 entries ---

Worst relative error, energy-force: 4.57e-07
Worst relative error, force-energy: 5.53e-07
Worst relative error, force-force: 5.91e-06

PASSED: all covariance blocks match finite differences of the true pipeline.
