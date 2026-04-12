import sys
import subprocess
import os
import importlib.util
import warnings
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning, module=r"arviz(\..*)?")


def configure_compiler_path():
    if os.name != "nt":
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    mingw_bin = os.path.join(conda_prefix, "Library", "mingw-w64", "bin")
    if os.path.isdir(mingw_bin):
        os.environ["PATH"] = mingw_bin + os.pathsep + os.environ.get("PATH", "")

    gpp_path = os.path.join(mingw_bin, "g++.exe")
    if not os.path.exists(gpp_path):
        os.environ["PYTENSOR_FLAGS"] = "cxx="


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "pymc": "pymc",
    "arviz": "arviz",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def bootstrap_environment(base_dir):
    req_file = os.path.join(base_dir, "requirements.txt")

    missing_packages = [
        package_name
        for package_name, import_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    if not missing_packages:
        return

    if os.path.exists(req_file):
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    req_file,
                ]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Dependency installation failed. Please check your network "
                "connection and pip availability."
            ) from exc
    else:
        raise FileNotFoundError(
            f"Required file not found: {req_file}. Cannot continue without dependencies."
        )

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    configure_compiler_path()

    bootstrap_environment(project_dir)

    from data_engine import load_and_preprocess, run_pca_fusion
    from bayesian_sim import run_mcmc_simulation, generate_posterior_predictions
    from regression_models import train_and_evaluate_regressors, build_univariate_relationship
    from visualizer import (
        plot_pca_variance,
        plot_mcmc_diagnostics,
        plot_prediction_band,
        plot_regression_predictions,
        plot_regression_metrics,
        plot_regression_coefficients,
        plot_univariate_linear_relationship,
        plot_3d_pca_scatter,
    )

    data_path = os.path.join(project_dir, "Occupancy_Estimation.csv")
    df_clean = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test, pca_model, feature_names = run_pca_fusion(df_clean)

    bayesian_model, trace = run_mcmc_simulation(X_train, y_train)
    mean_pred, lower_bound, upper_bound = generate_posterior_predictions(bayesian_model, trace, X_test, y_test)
    metrics_df, predictions, coefficients_df = train_and_evaluate_regressors(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )
    univariate_data = build_univariate_relationship(X_train, y_train, feature_names)

    plot_pca_variance(pca_model, save_dir=output_dir)
    plot_mcmc_diagnostics(trace, save_dir=output_dir)
    plot_prediction_band(y_test, mean_pred, lower_bound, upper_bound, save_dir=output_dir)
    plot_regression_predictions(y_test, predictions, save_dir=output_dir)
    plot_regression_metrics(metrics_df, save_dir=output_dir)
    plot_regression_coefficients(coefficients_df, save_dir=output_dir)
    plot_univariate_linear_relationship(univariate_data, save_dir=output_dir)

    # Prepare PCA data for 3D scatter plot
    pca_data = pd.DataFrame(
        pca_model.transform(df_clean[['S1_TEMP', 'S2_TEMP', 'S3_TEMP', 'S4_TEMP', 'S1_LIGHT', 'S2_LIGHT', 'S3_LIGHT', 'S4_LIGHT']]),
        columns=['MICROCLIMATE_PC1', 'MICROCLIMATE_PC2', 'MICROCLIMATE_PC3']
    )
    target = df_clean['ROOM_OCCUPANCY_COUNT']

    # Plot 3D PCA scatter
    plot_3d_pca_scatter(pca_data, target, save_dir=output_dir)

    print("\nRegression metrics (OLS/Ridge/Lasso):")
    print(metrics_df.to_string(index=False))
    print(f"Saved figures to: {output_dir}")
