import pymc as pm
import numpy as np


def run_mcmc_simulation(X_train, y_train):
    X_train_array = np.asarray(X_train, dtype=float)
    y_train_array = np.asarray(y_train, dtype=float)

    if not np.isfinite(X_train_array).all() or not np.isfinite(y_train_array).all():
        raise ValueError("Input data contains non-finite values.")
    
    with pm.Model() as bayesian_model:
        X_data = pm.Data('X_data', X_train_array)
        y_data = pm.Data('y_data', y_train_array)

        alpha = pm.Normal('alpha', mu=0, sigma=5)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=X_train_array.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=5)

        mu = alpha + pm.math.dot(X_data, beta)
        pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_data)

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.9,
            init='adapt_diag',
            random_seed=42,
            progressbar=True,
        )
        
    return bayesian_model, trace


def generate_posterior_predictions(bayesian_model, trace, X_test, y_test):
    X_test_array = np.asarray(X_test, dtype=float)
    y_test_array = np.asarray(y_test, dtype=float)
    with bayesian_model:
        pm.set_data({'X_data': X_test_array, 'y_data': y_test_array})
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['Y_obs'],
            random_seed=42,
            progressbar=False,
        )
        
    y_preds = ppc.posterior_predictive['Y_obs'].values
    y_preds_flat = y_preds.reshape(-1, y_preds.shape[-1])
    
    mean_pred = np.mean(y_preds_flat, axis=0)
    lower_bound = np.percentile(y_preds_flat, 2.5, axis=0)
    upper_bound = np.percentile(y_preds_flat, 97.5, axis=0)
    
    return mean_pred, lower_bound, upper_bound