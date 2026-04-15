# PolyU BSE2610 Group Project_Occupancy Estimation

PCA + Bayesian and traditional regression (OLS, Ridge, Lasso) for room occupancy prediction.

## Important Prerequisites for PyMC Acceleration

This project heavily relies on `PyMC` for Bayesian linear regression. To enable the C-backend acceleration via PyTensor (reducing MCMC sampling time from hours to seconds), **a C++ compiler (`g++`) is strictly required.**

### Windows Users:
- Ensure you have `g++` installed and correctly added to your system `PATH` via MSYS2 / MinGW-w64.
- Alternatively, use Conda: `conda install -c conda-forge m2w64-toolchain`

### macOS Users:
- Ensure Xcode Command Line Tools are installed: `xcode-select --install`

### Linux Users:
- Install `build-essential`: `sudo apt install build-essential`

*Note: Running this project without a properly configured C++ compiler will force PyMC to fallback to a pure-Python backend, resulting in extremely slow execution times (potentially 8+ hours).*

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Main Files

The project structure is as follows:

```plaintext
.
├── main.py
│   ├── configure_compiler_path()
│   ├── bootstrap_environment()
│   ├── data_engine.py
│   │   ├── load_and_preprocess()
│   │   ├── run_pca_fusion()
│   │   ├── run_mcmc_simulation()
│   │   └── generate_posterior_predictions()
│   ├── regression_models.py
│   │   ├── train_and_evaluate_regressors()
│   │   └── build_univariate_relationship()
│   └── visualizer.py
│       ├── plot_pca_variance()
│       ├── plot_mcmc_diagnostics()
│       ├── plot_prediction_band()
│       ├── plot_regression_predictions()
│       ├── plot_regression_metrics()
│       ├── plot_regression_coefficients()
│       ├── plot_univariate_linear_relationship()
│       └── plot_3d_pca_scatter()
```

## Output

Generated in `figures/`:

- **`fig1_pca_variance.png`**: PCA variance explained (individual + cumulative)
- **`fig2_mcmc_trace.png`**: Bayesian MCMC trace diagnostics
- **`fig3_posterior_density.png`**: Posterior density for top features
- **`fig4_prediction_band.png`**: Bayesian prediction band (95% CI)
- **`fig5_regression_predictions.png`**: OLS vs Ridge vs Lasso predictions
- **`fig6_regression_metrics.png`**: Regression performance metrics (RMSE, MAE, R²)
- **`fig7_regression_coefficients.png`**: Coefficient comparison (top features)
- **`fig8_two_variable_relationship.png`**: Single-feature regression (scatter + fits)
- **`fig9_3d_pca_scatter.png`**: 3D scatter of PCA components with occupancy
