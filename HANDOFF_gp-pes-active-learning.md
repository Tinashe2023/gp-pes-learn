# Handoff: Energy–Force GP Active Learning for Interatomic Potentials

Written 9 Nov 2026 to continue this work in a new chat. Read top to bottom before doing anything. Section 9 says what to do next.

---

## 1. Target and constraints

- **Venue:** AIP Publishing special topic **"AI for Computational Physical Science"** (guest editors Hongjun Xiang, Jian Sun, Fang Liu). Submission deadline **31 Jan 2027**, about 12 weeks from this handoff.
- **Do not confuse it with** *"Quantum Computing for Computational Physics: Algorithms, Simulations, and Applications"* (deadline 30 Nov 2026, editors Izmaylov, Barnes, Love). That is a separate call. It was analysed earlier and mixed up with this one once by mistake. The call text for this project is in `AIP.pdf` in the Project files; the quantum call is the other PDF.
- **Scope items this paper targets:** "Active Learning, Uncertainty Quantification, and Autonomous Computational Workflows", with a secondary link to "Data Infrastructure, Benchmarks, Reproducibility, and Open-Source Software" if code and comparisons are released.
- **User constraints:** no HPC, no special hardware, no collaborators required. Laptop-scale math, algorithm design, ML and simulation only. Title style: concrete and declarative, with no words like "New" or "Or" (the model example was an APL paper on a probe–suspended membrane resonance model for graphene).
- **User working style:** builds one small piece at a time, verifies it, then moves on. Wants code that has been verified before it is handed over. The user runs everything on **Windows**, inside a conda env, from the project root.

## 2. Current working title

**Energy–Force Gaussian-Process Active Learning with Information-Gain Convergence Guarantees for Interatomic-Potential Construction**

The earlier title "Regret-Bounded Gaussian-Process Bandit Acquisition…" was **retired**. It implied deriving the bound from scratch, and Section 3 shows that overclaims.

## 3. How the topic was chosen (literature review outcome)

Five initial candidates were checked by web search. **All five were already crowded, with 2025–2026 precedents:**

| Candidate | Finding |
|---|---|
| Equivariant GNN for point-defect formation energies in 2D materials | Already done several times in 2025 (DefiNet, persistent-homology GNNs, QPOD database) |
| Conformal / calibrated UQ for MLIPs | Active since 2022; an April 2026 paper does environment-dependent conformal intervals on MACE-MP-0 |
| Symbolic regression for thermal-transport descriptors | SISSO, GP-SR, SyMANTIC, GDS, Bayesian SISSO (June 2026) all exist |
| Active-learning acquisition functions for PES | Decade-long literature, still active in 2026 |
| Benchmarking protocols for MLIPs | Hottest niche: MLIPAudit, CatBench, FAIR Chemistry leaderboard, a June 2026 benchmark of 15 pretrained MLIPs |

Further checks found DFT/XC-functional learning, nonadiabatic-dynamics ML and AI-driven multiscale modelling also saturated.

**The gap that survived:** active learning for PES/MLIP construction in the chemistry/materials literature is essentially heuristic (uncertainty sampling, committee disagreement, D-optimality). It has no formal sample-efficiency guarantees. GP-bandit information-gain theory exists in the ML-theory community (Srinivas et al., Vakili et al. 2021).

**Late discovery that changed the framing:** Yu, Zimmer & Nguyen-Tuong (Bosch), *"Batch Active Learning in Gaussian Process Regression using Derivatives"* (arXiv 2408.01861, 2024). It already:
- proves derivative observations increase information gain,
- bounds maximum information gain for derivative-GP by extending Srinivas et al. (their Prop. 2 extends Lemma 7.6),
- proves a predictive-covariance decay rate O(log(t)^(d+1)/t) under D-, A- and E-optimality.

It was tested only on a toy 2D function, a fuel-injection system and elevation maps. It has no physics, no invariant kernels, and no link to the MLIP active-learning literature.

**Honest novelty claim, as it now stands:** the core theorem machinery is not new. The contribution is the **adaptation and bridge**:
1. symmetry-invariant SOAP-based atom-summed kernels,
2. joint energy+force observations through a descriptor Jacobian,
3. sequential, DFT-in-the-loop querying rather than fixed batches,
4. validation against the acquisition heuristics the MLIP field actually uses.

Yu et al. 2024 must be cited as the foundation being extended. "No paper found in search" is not proof of absence, so a final check of arXiv and citation trails should be repeated before submission.

**Working hypothesis (not verified):** the atom-summed kernel K(R,R') = Σ_p Σ_q k(D_p(R), D_q(R')) is an ordinary valid kernel on configuration space, and force observations are gradient observations in that space. If so, Yu et al.'s framework applies to the induced kernel. The open mathematical work would then be the **eigenvalue decay of the induced kernel**, which depends on the smoothness of the SOAP map and on symmetry reducing the intrinsic dimension. This is the most promising theory task. Check it carefully before relying on it.

## 4. Environment and project layout

- **Project root (user's PC):** `C:\Users\TinasheHando\Desktop\2026 Research\gp-pes-active-learning`
- **Conda env:** `gp-pes-active-learning` (Python 3.11). Packages: numpy, scipy, matplotlib, pandas, sympy, jupyterlab, pytest, torch, gpytorch, ase, dscribe, tqdm, pyyaml. A pip `requirements.txt` equivalent exists. CPU only is fine.
- **Folders:** `literature/{papers,notes}`, `theory/{derivations,problem_setup.md}`, `src/{kernels,gp_models,acquisition,active_learning,descriptors,utils}`, `data/{raw,processed,toy_pes}`, `experiments/{configs,scripts,results}`, `notebooks`, `figures`, `paper/{manuscript,figures,supplementary}`, `tests`. Created by `setup_project.py`.
- The sandbox I used to verify code **resets between sessions**. Reinstall with `pip install --break-system-packages numpy scipy ase dscribe` if you need to run code there.

## 5. Data: rMD17

- **Download (verified working):** `https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1` (record page `https://archive.materialscloud.org/record/2020.82`). Size 1016.9 MiB, MD5 `cb1a927628d96f2e966025da4fb63d18`. The older `record/file?filename=…&record_id=466` URL is dead and returns a ~15 KB HTML page. Figshare renders blank without JavaScript.
- **Extracted to** `data/raw/rmd17/` (expected `npz_data/rmd17_<molecule>.npz`). The 512 GiB figure on the Materials Cloud page is cumulative download traffic, not dataset size.
- **Keys and units:** `nuclear_charges`, `coords` (Å), `energies` (**kcal/mol**), `forces` (**kcal/mol/Å**), plus `old_*` arrays from the original MD17 that should not be mixed in.
- **Ethanol, as loaded on the user's machine:** 9 atoms, 100,000 frames. Energy min −97086.973, max −97051.484, mean −97076.076, std 4.293 kcal/mol. Mean |F| per atom 40.459, max 231.202. Net force per frame is exactly 0.00000.
- **Implications for the paper:**
  - Energies are not zero-centered, so subtract the mean (and probably standardize) before GP fitting.
  - The trajectory stays close to equilibrium, so a random split will not create a meaningful extrapolation gap. Split by a structural coordinate (bond length, dihedral) or by contiguous trajectory segments.
  - The rMD17 documentation advises not training on more than about 1000 frames, since frames are correlated.

## 6. Verified results so far

| Step | Result |
|---|---|
| Toy derivative-GP, `sin(x)` | Fit accurate (RMSE 0.0054 / 0.0070) but gap variance only about 1.2× dense variance, so it was a weak test |
| Toy derivative-GP v2 (multi-frequency, gap x∈[3,8]) | Dense RMSE 0.0076, gap RMSE 1.035, variance ratio **1873×**, learned lengthscale 0.527, true curve inside the confidence band |
| rMD17 ethanol load | Shapes, units and zero net force confirmed (see Section 5) |
| dscribe SOAP analytical vs numerical derivatives | Relative error 2.1×10⁻⁹. SOAP used `r_cut=5.0, n_max=4, l_max=3`, **no `weighting` argument**, giving descriptor shape (9, 312) and derivative shape (9, 9, 3, 312) |
| **dscribe self-term bug** | See below |
| Chain-rule kernel blocks vs finite differences (water, `r_cut=5, n_max=3, l_max=2`, lengthscale 3, variance 1) | Worst relative error: energy–force 4.6×10⁻⁷, force–energy 5.5×10⁻⁷, force–force 5.9×10⁻⁶. All 9 + 9 + 81 entries checked |

**dscribe self-term bug.** `soap.derivatives()` returns **zeros** for `der[c, c]` (a center atom's descriptor differentiated with respect to its own position), in both analytical and numerical modes. The array convention is `der[center, moved_atom, xyz, feature]`. Because SOAP is translation-invariant, the correct value is `der[c, c] = −Σ_{j≠c} der[c, j]`, which is implemented as `fix_self_terms`. After the fix, the whole Jacobian matched finite differences to 8×10⁻⁹. The earlier analytical-vs-numerical check was blind to this because both modes are wrong in the same way. **Any code that uses dscribe derivatives must apply this fix.** It is worth one sentence in the paper's methods. A separate known dscribe issue (#113) says analytical derivatives are wrong when a smooth `weighting` function is used, which is why `weighting` is deliberately left unset.

## 7. The model (as implemented)

- Total energy E(R) = Σ_i ε(D_i(R)) with ε ~ GP(0, k), where k is an RBF kernel on the SOAP descriptor D_i of atom i. Forces F = −∂E/∂R.
- With `Jb[q, m, k, :] = ∂D_q^b/∂R^b_{m,k}`:
  - Cov(E^a, E^b) = Σ_{p,q} k(D^a_p, D^b_q)
  - Cov(E^a, F^b_{m,k}) = −Σ_{p,q} ∇_{D2}k · Jb[q,m,k]
  - Cov(F^a_{m,k}, E^b) = −Σ_{p,q} ∇_{D1}k · Ja[p,m,k], with ∇_{D1}k = −∇_{D2}k
  - Cov(F^a_{m,k}, F^b_{n,l}) = Σ_{p,q} Ja[p,m,k]ᵀ · [k(I/ℓ² − uuᵀ/ℓ⁴)] · Jb[q,n,l], with u = D^a_p − D^b_q
- Open modelling choices not yet examined:
  - A single RBF kernel is shared across all species.
  - Hyperparameters are not yet learned.
  - SOAP parameters for ethanol have not been tuned.

## 8. Files delivered (with intended locations)

| File | Goes in |
|---|---|
| `setup_project.py` | anywhere; run once |
| `environment.yml`, `requirements.txt` | project root |
| `sanity_check_derivative_gp.py` (v2) | `tests/` |
| `rmd17.py` | `src/utils/` |
| `load_ethanol.py` | `experiments/scripts/` |
| `check_soap_derivatives.py` | `experiments/scripts/` |
| `soap_ef_kernel.py` | `src/kernels/` (verified; includes `fix_self_terms`, `soap_with_jacobian`, and the four covariance blocks) |
| `test_soap_ef_kernel.py` | `tests/` (run from project root; verified in my sandbox, **not yet confirmed on the user's machine**) |

Attach `soap_ef_kernel.py` and `test_soap_ef_kernel.py` (and `rmd17.py` if useful) to the new chat, because the sandbox does not keep files.

## 9. Next steps

1. **Fit script on ethanol.** Assemble the full training covariance for N frames (28 rows per frame: 1 energy + 27 forces). Centre and standardize energies. Add observation noise for energies and forces. Learn lengthscale, variance and noise by maximizing marginal likelihood (numpy/scipy or torch). Produce a **learning curve** of force and energy error against N on held-out frames. Cost notes: N=100 gives a 2,800×2,800 matrix; each frame's Jacobian is about 0.6 MB, so around 600 MB at N=1000.
2. **Tune SOAP** (`r_cut`, `n_max`, `l_max`) and check whether species-specific kernels matter.
3. **Design the gap experiment** on ethanol using a structural-coordinate split, plus an analytic toy PES (Müller–Brown, double well) in `data/toy_pes/`.
4. **Active-learning loop and baselines:** random, max-variance (uncertainty sampling), committee disagreement, D-optimality, and the information-gain acquisition from the theory.
5. **Theory:** state the problem formally in `theory/problem_setup.md`. Work out the information-gain bound and covariance decay rate for the induced configuration-space kernel (Section 3 hypothesis). Check the proof against numerical experiments.
6. **Before committing to writing:** repeat the arXiv and citation-trail novelty check for "information gain / regret / sample complexity" with derivative or force observations in MLIP active learning.
7. **Suggested rough schedule to 31 Jan 2027 (not agreed):** about 2 weeks for fit and learning curve; weeks 3–5 for the AL loop and baselines; weeks 4–8 for theory; weeks 8–10 for final experiments and figures; last 2 weeks for writing and buffer.

## 10. Key references

- Yu, Zimmer, Nguyen-Tuong (2024). Batch Active Learning in Gaussian Process Regression using Derivatives. arXiv:2408.01861
- Srinivas, Krause, Kakade, Seeger (2010/2012). Gaussian Process Optimization in the Bandit Setting (GP-UCB)
- Vakili, Khezeli, Picheny (2021). On Information Gain and Regret Bounds in Gaussian Process Bandits. arXiv:2009.06966
- Relative Information Gain and Gaussian Process Regression. arXiv:2510.04277
- Scaling Gaussian Process Regression with Full Derivative Observations. arXiv:2505.09134
- Himanen et al. (2020) and Updates to the DScribe Library (arXiv:2303.14046); dscribe GitHub issue #113
- Christensen & von Lilienfeld (2020). rMD17, on the role of gradients for ML of molecular energies and forces
- Bartók & Csányi, Gaussian Approximation Potentials (the same chain-rule kernel structure, used in practice)
