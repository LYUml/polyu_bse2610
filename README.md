# PolyU BSE2610 Group Project_Occupancy Estimation

PCA + Bayesian linear regression (PyMC) for room occupancy prediction.

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
