"""Test unitari per src/graphs/network_builders.py"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graphs.network_builders import (
    correlation_mst,
    correlation_pmfg,
    granger_causality_network,
    diebold_yilmaz_network,
    sector_network,
    build_stage_weight_matrices,
)


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(0)
    n, T = 6, 200
    factor = rng.normal(0, 1, T)
    tickers = [f"A{i}" for i in range(n)]
    data = {}
    for i, t in enumerate(tickers):
        idio = rng.normal(0, 1, T)
        data[t] = 0.5 * factor + 0.5 * idio
    dates = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(data, index=dates)


def test_correlation_mst_is_tree(synthetic_returns):
    mst = correlation_mst(synthetic_returns)
    assert mst.number_of_nodes() == 6
    assert mst.number_of_edges() == 5  # N-1 archi per un albero
    assert nx.is_tree(mst)


def test_correlation_pmfg_is_planar_and_bounded(synthetic_returns):
    pmfg = correlation_pmfg(synthetic_returns)
    is_planar, _ = nx.check_planarity(pmfg)
    assert is_planar
    n = pmfg.number_of_nodes()
    assert pmfg.number_of_edges() <= 3 * n - 6


def test_granger_network_is_directed(synthetic_returns):
    g = granger_causality_network(synthetic_returns, max_lag=2, alpha=0.10)
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 6


def test_diebold_yilmaz_rows_sum_to_one(synthetic_returns):
    graph, matrix, total_index = diebold_yilmaz_network(synthetic_returns, var_lag=1, horizon=5)
    row_sums = matrix.sum(axis=1).values
    np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-6)
    assert 0.0 <= total_index <= 100.0


def test_sector_network_connects_same_sector_only():
    tickers = ["A", "B", "C", "D"]
    sector_map = {"A": "Tech", "B": "Tech", "C": "Fin", "D": "Fin"}
    g = sector_network(tickers, sector_map)
    assert g.has_edge("A", "B")
    assert g.has_edge("C", "D")
    assert not g.has_edge("A", "C")


def test_stage_weight_matrices_row_stochastic(synthetic_returns):
    mst = correlation_mst(synthetic_returns)
    nodes = list(synthetic_returns.columns)
    matrices = build_stage_weight_matrices(mst, nodes, max_stage=2, weighting="equal")
    assert len(matrices) == 2
    for S in matrices:
        row_sums = S.sum(axis=1)
        # ogni riga somma a 1 se il nodo ha almeno un vicino a quello stage, altrimenti 0
        assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))
