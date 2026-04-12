import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os
from mpl_toolkits.mplot3d import Axes3D

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

def plot_regression_predictions(y_test, predictions, samples_to_show=100, save_dir=None):
    n_show = min(samples_to_show, len(y_test))
    idx = np.arange(n_show)
    y_true = np.asarray(y_test)[:n_show]

    plt.figure(figsize=(12, 5))
    plt.plot(idx, y_true, "ko", label="Ground Truth", markersize=4)

    style_map = {
        "Ridge": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2.6, "zorder": 2},
        "Lasso": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.2, "zorder": 3},
        "OLS": {
            "color": "#1f77b4",
            "linestyle": "--",
            "linewidth": 2.2,
            "marker": "o",
            "markersize": 3,
            "markevery": 6,
            "zorder": 4,
        },
    }

    # Draw OLS last with markers so it is still visible if curves overlap.
    for model_name in ["Ridge", "Lasso", "OLS"]:
        if model_name in predictions:
            plt.plot(
                idx,
                np.asarray(predictions[model_name])[:n_show],
                label=f"{model_name} Prediction",
                **style_map[model_name],
            )

    plt.title("Fig 5: OLS vs Ridge vs Lasso Predictions", fontsize=14)
    plt.xlabel("Test Set Sample Time-Step Index", fontsize=12)
    plt.ylabel("Room Occupancy Count", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    _save_current_figure(save_dir, "fig5_regression_predictions.png")
    plt.show()

def plot_regression_metrics(metrics_df, save_dir=None):
    metric_plot = metrics_df.set_index("Model")[["RMSE", "MAE", "R2"]]

    plt.figure(figsize=(10, 5))
    metric_plot.plot(kind="bar", ax=plt.gca(), width=0.75)
    plt.title("Fig 6: Regression Performance Metrics Comparison", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(loc="best")
    plt.tight_layout()
    _save_current_figure(save_dir, "fig6_regression_metrics.png")
    plt.show()

def plot_regression_coefficients(coefficients_df, top_k=12, save_dir=None):
    coef_abs_max = coefficients_df.abs().max(axis=1)
    selected_features = coef_abs_max.sort_values(ascending=False).head(top_k).index
    plot_df = coefficients_df.loc[selected_features]

    plt.figure(figsize=(11, 6))
    plot_df.plot(kind="bar", ax=plt.gca(), width=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Fig 7: Coefficient Comparison (Top Features)", fontsize=14)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Coefficient Value", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best")
    plt.tight_layout()
    _save_current_figure(save_dir, "fig7_regression_coefficients.png")
    plt.show()

def plot_univariate_linear_relationship(univariate_data, save_dir=None):
    x_train = np.asarray(univariate_data["x_train"])
    y_train = np.asarray(univariate_data["y_train"])
    x_grid = np.asarray(univariate_data["x_grid"])
    lines = univariate_data["lines"]
    feature_name = str(univariate_data["feature_name"])

    plt.figure(figsize=(10, 6))
    plt.scatter(
        x_train,
        y_train,
        color="#4C72B0",
        alpha=0.28,
        s=18,
        edgecolor="none",
        label="Training Samples",
    )

    style_map = {
        "Ridge": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2.6, "zorder": 2},
        "Lasso": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.2, "zorder": 3},
        "OLS": {
            "color": "#1f77b4",
            "linestyle": "--",
            "linewidth": 2.2,
            "marker": "o",
            "markersize": 3,
            "markevery": 15,
            "zorder": 4,
        },
    }

    for model_name in ["Ridge", "Lasso", "OLS"]:
        if model_name in lines:
            plt.plot(
                x_grid,
                np.asarray(lines[model_name]),
                label=f"{model_name} Fit",
                **style_map[model_name],
            )

    plt.title(f"Fig 8: Two-Variable Linear Relationship ({feature_name} vs Occupancy)", fontsize=14)
    plt.xlabel(f"{feature_name} (Standardized)", fontsize=12)
    plt.ylabel("Room Occupancy Count", fontsize=12)
    plt.legend(loc="best")
    plt.tight_layout()
    _save_current_figure(save_dir, "fig8_two_variable_relationship.png")
    plt.show()

def plot_3d_pca_scatter(pca_data, target, save_dir=None):
    """
    Plot a 3D scatter plot of the first three principal components (PC1, PC2, PC3).

    Parameters:
        pca_data (pd.DataFrame): DataFrame containing PC1, PC2, PC3 as columns.
        target (pd.Series): Target variable to color the points.
        save_dir (str): Directory to save the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        pca_data['MICROCLIMATE_PC1'],
        pca_data['MICROCLIMATE_PC2'],
        pca_data['MICROCLIMATE_PC3'],
        c=target, cmap='viridis', s=20, alpha=0.8
    )

    ax.set_title('Fig 9: 3D Scatter of PCA Components', fontsize=14)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Room Occupancy Count', fontsize=12)

    plt.tight_layout()
    if save_dir:
        _save_current_figure(save_dir, 'fig9_3d_pca_scatter.png')
    plt.show()