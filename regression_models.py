import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate_regressors(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    ridge_alpha=1.0,
    lasso_alpha=0.05,
):
    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=ridge_alpha),
        "Lasso": Lasso(alpha=lasso_alpha, random_state=42, max_iter=10000),
    }

    predictions = {}
    metric_rows = []
    coefficient_map = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        metric_rows.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

        coefficient_map[model_name] = model.coef_

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    coefficients_df = pd.DataFrame(coefficient_map, index=feature_names)

    return metrics_df, predictions, coefficients_df


def build_univariate_relationship(
    X_train,
    y_train,
    feature_names,
    ridge_alpha=1.0,
    lasso_alpha=0.05,
):
    X_train_array = np.asarray(X_train, dtype=float)
    y_train_array = np.asarray(y_train, dtype=float)

    if X_train_array.ndim != 2 or X_train_array.shape[1] == 0:
        raise ValueError("X_train must be a 2D array with at least one feature.")

    correlations = []
    for i in range(X_train_array.shape[1]):
        corr = np.corrcoef(X_train_array[:, i], y_train_array)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        correlations.append(abs(corr))

    selected_index = int(np.argmax(correlations))
    selected_feature_name = str(feature_names[selected_index])

    x_train_1d = X_train_array[:, selected_index].reshape(-1, 1)
    x_grid = np.linspace(x_train_1d.min(), x_train_1d.max(), 200).reshape(-1, 1)

    uni_models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=ridge_alpha),
        "Lasso": Lasso(alpha=lasso_alpha, random_state=42, max_iter=10000),
    }

    lines = {}
    for model_name, model in uni_models.items():
        model.fit(x_train_1d, y_train_array)
        lines[model_name] = model.predict(x_grid)

    return {
        "feature_name": selected_feature_name,
        "x_train": x_train_1d.ravel(),
        "y_train": y_train_array,
        "x_grid": x_grid.ravel(),
        "lines": lines,
    }