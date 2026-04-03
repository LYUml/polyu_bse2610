import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 120


def _save_current_figure(save_dir, filename):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')


def plot_pca_variance(pca, save_dir=None):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 4), explained_variance, alpha=0.6, color='#4C72B0', label='Individual Variance')
    plt.step(range(1, 4), cumulative_variance, where='mid', color='#C44E52', label='Cumulative Variance')
    plt.axhline(y=0.85, color='k', linestyle='--', label='85% Threshold')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('Fig 1: PCA Variance of Spatial Microclimate Sensors')
    plt.legend()
    plt.tight_layout()
    _save_current_figure(save_dir, 'fig1_pca_variance.png')
    plt.show()


def plot_mcmc_diagnostics(trace, save_dir=None):
    az.plot_trace(trace, var_names=['alpha', 'sigma'], compact=True)
    plt.suptitle('Fig 2: MCMC Trace Plot (Sampling Diagnostics)', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_current_figure(save_dir, 'fig2_mcmc_trace.png')
    plt.show()
    
    az.plot_posterior(trace, var_names=['beta'], coords={"beta_dim_0": [0, 1, 2, 3]}, hdi_prob=0.95)
    plt.suptitle('Fig 3: Posterior Density for Top Features (95% HDI)', fontsize=14, y=1.05)
    plt.tight_layout()
    _save_current_figure(save_dir, 'fig3_posterior_density.png')
    plt.show()


def plot_prediction_band(y_test, mean_pred, lower_bound, upper_bound, samples_to_show=100, save_dir=None):
    n_show = min(samples_to_show, len(y_test), len(mean_pred), len(lower_bound), len(upper_bound))

    plt.figure(figsize=(12, 5))
    idx = np.arange(n_show)
    plt.plot(idx, y_test.values[:n_show], 'ko', label='Ground Truth (Actual Occupancy)', markersize=4)
    plt.plot(idx, mean_pred[:n_show], color='#4C72B0', label='Bayesian Mean Prediction', linewidth=2)
    plt.fill_between(idx, lower_bound[:n_show], upper_bound[:n_show], color='#4C72B0', alpha=0.25,
                     label='95% Predictive Confidence Interval')
    
    plt.title('Fig 4: Probabilistic Prediction Band (95% CI) vs. Ground Truth', fontsize=14)
    plt.xlabel('Test Set Sample Time-Step Index', fontsize=12)
    plt.ylabel('Room Occupancy Count', fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    _save_current_figure(save_dir, 'fig4_prediction_band.png')
    plt.show()