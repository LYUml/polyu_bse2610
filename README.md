# PolyU BSE2610 Group Project_Occupancy Estimation

PCA + Bayesian linear regression (PyMC) for room occupancy prediction.

## Important Prerequisites for PyMC Acceleration

This project heavily relies on `PyMC` for Bayesian linear regression. To enable the C-backend acceleration via PyTensor (reducing MCMC sampling time from hours to seconds), **a C++ compiler (`g++`) is strictly required.**

**Windows Users:**
Ensure you have `g++` installed and correctly added to your system `PATH` via MSYS2 / MinGW-w64. 
Alternatively, use Conda: `conda install -c conda-forge m2w64-toolchain`

**macOS Users:**
Ensure Xcode Command Line Tools are installed: `xcode-select --install`

**Linux Users:**
Install `build-essential`: `sudo apt install build-essential`

*Note: Running this project without a properly configured C++ compiler will force PyMC to fallback to a pure-Python backend, resulting in extremely slow execution times (potentially 8+ hours).*

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Main Files

- `main.py`: run full pipeline
- `data_engine.py`: preprocessing + PCA
- `bayesian_sim.py`: Bayesian modeling + posterior prediction
- `visualizer.py`: figure plotting

## Output

Generated in `figures/`:

- `fig1_pca_variance.png`
- `fig2_mcmc_trace.png`
- `fig3_posterior_density.png`
- `fig4_prediction_band.png`
