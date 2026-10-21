"""Split-conformal prediction intervals for the price model.

We calibrate on a held-out slice the model never trained on. The score is the
absolute log ratio |log(actual) - log(predicted)|, so the interval is
multiplicative: [pred / e^q, pred * e^q]. That suits prices (positive,
right-skewed, error grows with price) and the lower bound can never go
negative. If the calibration and test listings are exchangeable, the
interval covers the true price at least 1 - alpha of the time, on average.
It is a marginal guarantee, not a per-listing one.
"""
import math

import numpy as np


def log_ratio_scores(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true, dtype=float), 1.0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1.0)
    return np.abs(np.log(y_true) - np.log(y_pred))


def conformal_quantile(scores, alpha):
    """The ceil((n + 1) * (1 - alpha))-th smallest score."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score.")
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(scores[k - 1])


def prediction_interval(y_pred, q):
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1.0)
    return y_pred * math.exp(-q), y_pred * math.exp(q)


def empirical_coverage(y_true, low, high):
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= low) & (y_true <= high)))
