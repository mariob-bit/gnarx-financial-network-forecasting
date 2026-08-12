"""Test unitari per src/models/gnar.py e src/models/variance_models.py"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graphs.network_builders import correlation_mst, build_stage_weight_matrices
from src.models.gnar import GNAR
from src.models.variance_models import GNGARCH, DSTARCH


@pytest.fixture
def toy_graph_and_data():
    rng = np.random.default_rng(1)
    n, T = 5, 300
    tickers = [f"A{i}" for i in range(n)]
    dates = pd.bdate_range("2019-01-01", periods=T)

    G = nx.cycle_graph(n)
    nx.set_edge_attributes(G, 1.0, "weight")
    G = nx.relabel_nodes(G, {i: tickers[i] for i in range(n)})
    S = build_stage_weight_matrices(G, tickers, max_stage=1, weighting="equal")[0]

    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = 0.1 * Y[t - 1] + 0.2 * (S @ Y[t - 1]) + rng.normal(0, 0.01, n)
    returns = pd.DataFrame(Y, index=dates, columns=tickers)

    exog = pd.DataFrame({"rate": np.linspace(0, 1, T)}, index=dates)
    return returns, exog, G, tickers


def test_gnar_fit_predict_shapes(toy_graph_and_data):
    returns, exog, G, tickers = toy_graph_and_data
    stage = build_stage_weight_matrices(G, tickers, max_stage=1, weighting="equal")

    model = GNAR(P=1, R=[1], stage_matrices=stage, exog_lags=[1])
    result = model.fit(returns, exog=exog)

    assert result.fitted.shape[1] == len(tickers)
    assert not np.isnan(result.r_squared)
    assert len(result.coefficients) == len(result.coef_names)

    pred = model.predict_one_step(returns.iloc[-3:], exog.iloc[-3:])
    assert list(pred.index) == tickers
    assert pred.notna().all()


def test_gnar_global_coefficients_shared_across_nodes(toy_graph_and_data):
    """Verifica che il modello GNAR globale abbia UN solo set di coefficienti
    (non N*P*(1+R)), coerente con la formulazione a parametri condivisi."""
    returns, exog, G, tickers = toy_graph_and_data
    stage = build_stage_weight_matrices(G, tickers, max_stage=1, weighting="equal")
    model = GNAR(P=2, R=[1, 1], stage_matrices=stage, exog_lags=[1])
    result = model.fit(returns, exog=exog)
    # P=2 -> own_lag1, net_lag1_r1, own_lag2, net_lag2_r1 = 4 feature + 1 esogena = 5
    assert len(result.coef_names) == 5


def test_gngarch_fit_produces_positive_variance(toy_graph_and_data):
    returns, exog, G, tickers = toy_graph_and_data
    stage = build_stage_weight_matrices(G, tickers, max_stage=1, weighting="equal")
    gngarch = GNGARCH(R=1, distribution="normal")
    result = gngarch.fit(returns, stage)
    assert (result.conditional_variance.values > 0).all()
    h_next = gngarch.forecast_one_step(returns.iloc[-1].values, result.conditional_variance.iloc[-1].values)
    assert np.all(h_next > 0)


def test_dst_arch_fit_produces_positive_variance(toy_graph_and_data):
    returns, exog, G, tickers = toy_graph_and_data
    W = nx.to_numpy_array(G, nodelist=tickers, weight="weight")
    dst = DSTARCH(ar_order=1)
    result = dst.fit(returns, W)
    assert (result.conditional_variance.values > 0).all()
    h_next = dst.forecast_one_step(returns.iloc[-1].values)
    assert np.all(h_next > 0)
    # gli autovalori di una matrice simmetrica riga-normalizzata sono reali e in [-1,1]
    assert np.all(np.abs(result.eigenvalues) <= 1.0 + 1e-8)
