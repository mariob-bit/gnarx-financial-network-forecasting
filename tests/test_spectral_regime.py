"""Test unitari per src/graphs/spectral_regime.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graphs.spectral_regime import (
    marchenko_pastur_bounds,
    rolling_eigen_spectrum,
    compute_spectral_regime_features,
)


def test_marchenko_pastur_bounds_sane():
    lo, hi = marchenko_pastur_bounds(n_assets=10, n_obs=200)
    assert 0 <= lo < 1
    assert hi > 1
    # per N=T, il bordo inferiore deve essere ~0
    lo2, hi2 = marchenko_pastur_bounds(n_assets=100, n_obs=100)
    assert lo2 < 1e-6


def _make_returns(n=8, T=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(rng.normal(0, 0.01, (T, n)), index=dates, columns=[f"A{i}" for i in range(n)])


def test_lambda1_share_in_valid_range():
    returns = _make_returns()
    spectrum = rolling_eigen_spectrum(returns, window=60)
    valid = spectrum["lambda1_share"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all() and (valid <= 1.0 + 1e-8).all()
    # con N asset indipendenti, lambda1_share atteso vicino a 1/N (nessun fattore dominante)
    assert valid.mean() < 0.6


def test_lambda1_share_rises_with_injected_common_factor():
    """Un fattore comune con loading elevato deve alzare chiaramente lambda1_share
    rispetto al caso puramente idiosincratico (validazione del segnale di regime)."""
    rng = np.random.default_rng(1)
    n, T = 8, 300
    dates = pd.bdate_range("2020-01-01", periods=T)
    idio = rng.normal(0, 0.01, (T, n))
    common = rng.normal(0, 0.01, T)
    stressed = 0.2 * idio + 0.95 * common[:, None]
    returns = pd.DataFrame(stressed, index=dates, columns=[f"A{i}" for i in range(n)])

    spectrum_stress = rolling_eigen_spectrum(returns, window=60)
    spectrum_normal = rolling_eigen_spectrum(_make_returns(n=n, T=T), window=60)

    assert spectrum_stress["lambda1_share"].dropna().mean() > spectrum_normal["lambda1_share"].dropna().mean()


def test_compute_spectral_regime_features_shape_and_hmm():
    returns = _make_returns(T=400)
    result = compute_spectral_regime_features(returns, window=60, roc_lag=5, fit_hmm=True)
    feats = result.features
    assert set(["lambda1_share", "lambda1_share_roc", "n_significant_mp", "regime_prob_stress"]).issubset(feats.columns)
    prob = feats["regime_prob_stress"].dropna()
    assert len(prob) > 0
    assert ((prob >= 0) & (prob <= 1)).all()


def test_compute_spectral_regime_features_without_hmm():
    returns = _make_returns(T=150)
    result = compute_spectral_regime_features(returns, window=60, roc_lag=5, fit_hmm=False)
    assert result.features["regime_prob_stress"].isna().all()
