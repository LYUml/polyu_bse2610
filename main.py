import sys
import subprocess
import os
import importlib.util
import warnings


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
    from visualizer import plot_pca_variance, plot_mcmc_diagnostics, plot_prediction_band

    data_path = os.path.join(project_dir, "Occupancy_Estimation.csv")
    df_clean = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test, pca_model, feature_names = run_pca_fusion(df_clean)

    bayesian_model, trace = run_mcmc_simulation(X_train, y_train)
    mean_pred, lower_bound, upper_bound = generate_posterior_predictions(bayesian_model, trace, X_test, y_test)

    plot_pca_variance(pca_model, save_dir=output_dir)
    plot_mcmc_diagnostics(trace, save_dir=output_dir)
    plot_prediction_band(y_test, mean_pred, lower_bound, upper_bound, save_dir=output_dir)

    print(f"Saved figures to: {output_dir}")
