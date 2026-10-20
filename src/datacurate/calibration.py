"""Isotonic calibration of judge scores to the human 1-5 scale.

Isotonic regression is monotonic, so it cannot improve rank correlation.
Its job is the scale: we report MAE and ECE against human scores before and after.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(x, y) -> IsotonicRegression:
    return IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip").fit(x, y)


def _equal_mass_bins(pred: np.ndarray, n_bins: int) -> list[np.ndarray]:
    return np.array_split(np.argsort(pred, kind="stable"), n_bins)


def ece(pred, target, n_bins: int = 10) -> float:
    """Expected calibration error for a regression score: mean |E[pred] - E[target]| per bin."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(
        sum(len(b) * abs(pred[b].mean() - target[b].mean()) for b in _equal_mass_bins(pred, n_bins))
        / len(pred)
    )


def mae(pred, target) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(target, float))))


def reliability_table(pred, target, n_bins: int = 10) -> pd.DataFrame:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return pd.DataFrame(
        [
            {"mean_pred": pred[b].mean(), "mean_human": target[b].mean(), "n": len(b)}
            for b in _equal_mass_bins(pred, n_bins)
        ]
    )
