"""
metrics.py — Forecast evaluation metrics.

Extracted from notebooks/03_forecasting.ipynb (section 6.1, compute_metrics).
Provides the four standard time-series forecasting metrics used to rank models
in the ensemble construction and to report CV performance.

Functions
---------
compute_metrics : MAE, RMSE, MAPE, SMAPE for a pair of actual/predicted arrays.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and SMAPE between actual and predicted arrays.

    All metrics are computed over the full array (pooled across series if the
    inputs span multiple series).  MAPE and SMAPE are returned as percentages.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions aligned to y_true.

    Returns
    -------
    dict with keys:
      MAE   — Mean Absolute Error (same units as y).
      RMSE  — Root Mean Squared Error (same units as y).
      MAPE  — Mean Absolute Percentage Error (percent, 0–∞).
      SMAPE — Symmetric MAPE (percent, 0–200).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_true - y_pred
    ae = np.abs(e)
    yt = np.abs(y_true)
    yp = np.abs(y_pred)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE": float((ae / np.clip(yt, 1e-8, None)).mean() * 100),
        "SMAPE": float((2 * ae / np.clip(yt + yp, 1e-8, None)).mean() * 100),
    }
