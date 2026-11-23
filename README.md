# Energy–Force Gaussian-Process Active Learning via Information-Gain Maximization for Potential Energy Surfaces

This repository contains the code, frozen data splits, figures, and tables for an energy–force Gaussian-process active-learning study of potential energy surfaces (PESs). The molecular benchmarks use the revised MD17 (rMD17) dataset for ethanol and malonaldehyde; analytic one- and two-dimensional benchmarks are included as well.

## Repository contents

- `src/`: SOAP energy–force kernels, exact GP model, and rMD17 loader.
- `experiments/scripts/`: benchmark, ablation, validation, and table-generation scripts.
- `experiments/configs/`: frozen rMD17 pool/test index splits and hyperparameter configuration.
- `tests/`: numerical checks for the GP model, kernel derivatives, and invariances.
- `figures/` and `tables/`: manuscript figures and tabulated results. Experiment JSON outputs and cached/processed data are generated locally and are not committed.

Run scripts from the repository root. The scripts use the settings and seeds defined in their source files; most do not accept command-line options.

## Set up the environment

Python 3.11 is used by the Conda environment file.

With Conda or Mamba:

```bash
conda env create -f environment.yml
conda activate gp-pes-active-learning
```

With Python's built-in virtual environment and pip:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements list the needed libraries but does not pin exact versions. The Conda file specifies Python 3.11; package versions may resolve to newer compatible releases over time.

## Download rMD17

The rMD17 archive is approximately 1 GiB and is not stored in this repository. Download it from the [Materials Cloud rMD17 record](https://archive.materialscloud.org/records/pfffs-fff86):

```bash
mkdir -p data/raw
curl -L -o data/raw/rmd17.tar.bz2 "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1"
tar -xjf data/raw/rmd17.tar.bz2 -C data/raw
```

On Windows PowerShell, use:

```powershell
New-Item -ItemType Directory -Force data/raw
Invoke-WebRequest -Uri "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1" -OutFile data/raw/rmd17.tar.bz2
tar -xjf data/raw/rmd17.tar.bz2 -C data/raw
```

The extracted files should include `data/raw/rmd17/npz_data/rmd17_ethanol.npz` and `rmd17_malonaldehyde.npz`. The loader in `src/utils/rmd17.py` reads the revised `energies` and `forces` fields (kcal/mol and kcal/mol/Å); it deliberately does not use the legacy `old_energies` or `old_forces` fields. To verify the downloaded archive, its MD5 checksum is `cb1a927628d96f2e966025da4fb63d18` (`md5sum data/raw/rmd17.tar.bz2` on macOS/Linux; `Get-FileHash data/raw/rmd17.tar.bz2 -Algorithm MD5` in PowerShell).

## Reproduce the benchmarks

Start with the analytic examples, which do not need rMD17:

```bash
python experiments/scripts/analytic_1d_benchmark.py
python experiments/scripts/toy_2d_active_learning.py
```

After downloading and extracting rMD17, run the cross-split and cross-molecule molecular benchmark:

```bash
python experiments/scripts/tier_c14_robustness_and_malonaldehyde.py
```

Other experiments are implemented as separate scripts in `experiments/scripts/`, including the acquisition, batch-size, derivative, and computational-scaling studies. The frozen pool/test indices are in `experiments/configs/`; `experiments/scripts/generate_splits.py` regenerates the standard splits from rMD17. Some experiments can take substantial time because they build dense energy–force covariance matrices.

The experiment scripts write intermediate JSON results to the ignored `experiments/results/` directory and plots to `figures/`. Where the needed experiment JSON files exist, generate the CSV and LaTeX tables with:

```bash
python experiments/scripts/generate_manuscript_tables.py
```

Run the automated checks from the repository root with `pytest`. These checks use small synthetic examples and do not require downloading rMD17.

The rMD17 trajectories are time-correlated. The dataset authors caution against treating large numbers of consecutive frames as independent training examples; consult the dataset paper and use the committed split indices when reproducing the reported comparisons.

## Data references

The molecular trajectories originate from MD17; the revised energies and forces used here are from rMD17. Please cite the dataset and its associated work when using these data:

- Christensen, A. S., & von Lilienfeld, O. A. (2020). On the role of gradients for machine learning of molecular energies and forces. *Machine Learning: Science and Technology, 1*(4), 045018. [https://doi.org/10.1088/2632-2153/abba6f](https://doi.org/10.1088/2632-2153/abba6f)
- Christensen, A. S., & von Lilienfeld, O. A. (2020). Revised MD17 dataset. Materials Cloud Archive, record 2020.82. [https://doi.org/10.24435/materialscloud:wy-kn](https://doi.org/10.24435/materialscloud:wy-kn)
- Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, I., Schütt, K. T., & Müller, K.-R. (2017). Machine learning accurate energies and forces for density functional theory with high accuracy and low computational cost. *Science Advances, 3*(5), e1603015. [https://doi.org/10.1126/sciadv.1603015](https://doi.org/10.1126/sciadv.1603015)

The revised-MD17 article number is `045018` (not `045019`). The Materials Cloud record requests citing its DOI when referencing the archive.
