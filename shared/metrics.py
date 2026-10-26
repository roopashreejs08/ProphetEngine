
import numpy as np
from sklearn.metrics import root_mean_squared_error


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error for RUL predictions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Cannot calculate RMSE on empty inputs.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Inputs must contain only finite values.")

    return float(root_mean_squared_error(y_true, y_pred))


def calculate_nasa_score(y_true, y_pred):
    """
    Calculate the NASA C-MAPSS asymmetric scoring function.

    Positive error (overestimated RUL): denominator 10.
    Negative error (underestimated RUL): denominator 13.
    Lower score is better.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Cannot calculate NASA score on empty inputs.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Inputs must contain only finite values.")

    error = y_pred - y_true

    penalties = np.where(
        error < 0,
        np.expm1(-error / 13.0),
        np.expm1(error / 10.0),
    )

    return float(np.sum(penalties))
