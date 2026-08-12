"""Test unitari per src/evaluation/metrics.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import mean_forecast_metrics, variance_forecast_metrics, qlike_loss, var_coverage_test


def test_mean_forecast_metrics_perfect_prediction():
    idx = pd.bdate_range("2021-01-01", periods=50)
    y = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=idx)
    metrics = mean_forecast_metrics(y, y.copy())
    assert np.isclose(metrics.loc["__ALL__", "RMSE"], 0.0)
    assert np.isclose(metrics.loc["__ALL__", "Directional_Accuracy"], 1.0)


def test_mean_forecast_metrics_detects_error():
    idx = pd.bdate_range("2021-01-01", periods=10)
    y_true = pd.DataFrame({"A": [1.0] * 10}, index=idx)
    y_pred = pd.DataFrame({"A": [1.0 + 0.5] * 10}, index=idx)
    metrics = mean_forecast_metrics(y_true, y_pred)
    assert np.isclose(metrics.loc["A", "RMSE"], 0.5)
    assert np.isclose(metrics.loc["A", "MAE"], 0.5)


def test_qlike_loss_lower_for_better_forecast():
    rng = np.random.default_rng(0)
    true_var = 1.0
    realized = (rng.normal(0, np.sqrt(true_var), 5000)) ** 2
    good_forecast = np.full(5000, true_var)
    bad_forecast = np.full(5000, true_var * 5)
    assert qlike_loss(realized, good_forecast) < qlike_loss(realized, bad_forecast)


def test_variance_forecast_metrics_shape():
    idx = pd.bdate_range("2021-01-01", periods=30)
    r2 = pd.DataFrame({"A": np.abs(np.random.randn(30))}, index=idx)
    h = pd.DataFrame({"A": np.abs(np.random.randn(30)) + 0.1}, index=idx)
    out = variance_forecast_metrics(r2, h)
    assert "QLIKE" in out.columns
    assert "__ALL__" in out.index


def test_var_coverage_test_runs():
    idx = pd.bdate_range("2021-01-01", periods=200)
    rng = np.random.default_rng(2)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 200)}, index=idx)
    mu = pd.DataFrame({"A": np.zeros(200)}, index=idx)
    h = pd.DataFrame({"A": np.full(200, 0.0001)}, index=idx)
    out = var_coverage_test(r, mu, h, alpha=0.05)
    assert "kupiec_pvalue" in out.columns
    assert out.loc["A", "n_obs"] == 200
