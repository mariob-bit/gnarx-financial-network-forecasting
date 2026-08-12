"""
src/graphs/network_builders.py
================================
Implementa i metodi di costruzione della topologia del grafo descritti nel
documento di riferimento:

  - Metodi statistico-dinamici:
      * Rete di correlazione (Pearson/Spearman) filtrata con Minimum
        Spanning Tree (Mantegna, 1999).
      * Rete di correlazione filtrata con Planar Maximally Filtered Graph
        (PMFG), versione algoritmica basata su controllo di planarità
        incrementale (Tumminello et al., 2005).
      * Rete di causalità di Granger (grafo diretto).
      * Rete di Volatility Spillover di Diebold-Yilmaz, basata sulla
        Generalized Forecast Error Variance Decomposition (GFEVD) di un VAR.
  - Metodo economico-strutturale:
      * Rete basata su classificazione settoriale (GICS-like).

Espone inoltre `build_stage_weight_matrices`, che trasforma un qualunque
grafo (pesato, diretto o meno) nelle matrici riga-stocastiche S_r usate
dai modelli GNAR/GNARX/GNGARCH per aggregare i vicini di ordine r.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metodi statistico-dinamici basati su correlazione
# --------------------------------------------------------------------------

def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Distanza di Mantegna: d_ij = sqrt(2*(1-rho_ij)) in [0, 2]."""
    return np.sqrt(2.0 * (1.0 - corr.clip(-1, 1)))


def correlation_mst(returns: pd.DataFrame, method: str = "pearson") -> nx.Graph:
    """
    Rete a Minimum Spanning Tree costruita sulla distanza di correlazione
    (Mantegna, 1999). Ritorna un grafo NON diretto e connesso con N-1 archi,
    pesato con la correlazione originale (non la distanza) per uso nel GNAR.
    """
    corr = returns.corr(method=method)
    dist = _correlation_distance(corr)

    G_full = nx.Graph()
    G_full.add_nodes_from(corr.columns)
    for i, j in itertools.combinations(corr.columns, 2):
        G_full.add_edge(i, j, distance=dist.loc[i, j], weight=corr.loc[i, j])

    mst = nx.minimum_spanning_tree(G_full, weight="distance")
    logger.info("MST costruito: %d nodi, %d archi", mst.number_of_nodes(), mst.number_of_edges())
    return mst


def correlation_pmfg(returns: pd.DataFrame, method: str = "pearson") -> nx.Graph:
    """
    Planar Maximally Filtered Graph (Tumminello et al., 2005): si aggiungono
    iterativamente gli archi in ordine di distanza crescente, mantenendo
    l'arco solo se il grafo risultante resta planare, fino a raggiungere il
    numero massimo di archi di un grafo planare (3N - 6, N>=3).

    Implementazione via controllo di planarità incrementale di networkx
    (adeguata per reti di decine di nodi come quelle di questo progetto;
    per reti molto più grandi servirebbero algoritmi di embedding dedicati).
    """
    corr = returns.corr(method=method)
    dist = _correlation_distance(corr)
    nodes = list(corr.columns)
    n = len(nodes)
    max_edges = 3 * n - 6 if n >= 3 else n - 1

    edges_sorted = sorted(
        itertools.combinations(nodes, 2), key=lambda e: dist.loc[e[0], e[1]]
    )

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, j in edges_sorted:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(i, j, distance=dist.loc[i, j], weight=corr.loc[i, j])
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            G.remove_edge(i, j)

    logger.info("PMFG costruito: %d nodi, %d archi (max teorico %d)", n, G.number_of_edges(), max_edges)
    return G


# --------------------------------------------------------------------------
# Rete di causalità di Granger (grafo diretto)
# --------------------------------------------------------------------------

def granger_causality_network(returns: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> nx.DiGraph:
    """
    Costruisce un grafo diretto: arco j -> i se i valori passati di j
    migliorano significativamente (test F, p-value < alpha) la previsione
    di i, al ritardo che minimizza il p-value tra 1..max_lag.

    Il peso dell'arco è 1 - p_value (maggiore per relazioni più significative);
    i pesi vengono poi normalizzati per riga in `build_stage_weight_matrices`.
    """
    tickers = list(returns.columns)
    G = nx.DiGraph()
    G.add_nodes_from(tickers)

    for i, j in itertools.permutations(tickers, 2):
        # grangercausalitytests vuole [target, causa] nelle colonne
        data = returns[[i, j]].dropna()
        if len(data) < max_lag * 3:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = grangercausalitytests(data, maxlag=max_lag)
        except Exception:
            continue
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        best_p = min(pvals)
        if best_p < alpha:
            G.add_edge(j, i, weight=1.0 - best_p, pvalue=best_p, best_lag=int(np.argmin(pvals)) + 1)

    logger.info(
        "Rete di Granger costruita: %d nodi, %d archi diretti significativi (alpha=%.2f)",
        G.number_of_nodes(), G.number_of_edges(), alpha,
    )
    return G


# --------------------------------------------------------------------------
# Rete di Volatility Spillover di Diebold-Yilmaz (GFEVD da VAR)
# --------------------------------------------------------------------------

def _ma_coefficients_from_var(coefs: np.ndarray, horizon: int) -> List[np.ndarray]:
    """
    Calcola le matrici di rappresentazione Media Mobile Theta_h di un VAR(p),
    per h = 0..horizon-1, tramite la ricorsione:
        Theta_0 = I
        Theta_h = sum_{l=1}^{min(h,p)} A_l @ Theta_{h-l}
    `coefs` ha shape (p, N, N) (coefs[l-1] = A_l).
    """
    p, n, _ = coefs.shape
    thetas = [np.eye(n)]
    for h in range(1, horizon):
        theta_h = np.zeros((n, n))
        for l in range(1, min(h, p) + 1):
            theta_h += coefs[l - 1] @ thetas[h - l]
        thetas.append(theta_h)
    return thetas


def diebold_yilmaz_network(
    returns: pd.DataFrame, var_lag: int = 2, horizon: int = 10
) -> Tuple[nx.DiGraph, pd.DataFrame, float]:
    """
    Costruisce la rete di spillover di volatilità di Diebold & Yilmaz (2012):
      1. Stima un VAR(var_lag) sui rendimenti (proxy di shock informativo;
         in alternativa si può passare una serie di volatilità realizzata).
      2. Calcola la Generalized FEVD a orizzonte `horizon`.
      3. Normalizza per riga (ogni riga somma a 1): theta_ij = quota della
         varianza dell'errore di previsione di i spiegata da shock a j.

    Ritorna (grafo diretto pesato, matrice di spillover N x N, total spillover index).
    Il grafo ha un arco j -> i con peso theta_ij (j "spiega"/contagia i).
    """
    tickers = list(returns.columns)
    n = len(tickers)
    model = VAR(returns.values)
    fitted = model.fit(var_lag)
    coefs = fitted.coefs  # (p, n, n)
    sigma = fitted.sigma_u  # (n, n) covarianza residui

    thetas = _ma_coefficients_from_var(coefs, horizon)
    sigma_diag = np.diag(sigma)

    numer = np.zeros((n, n))
    denom = np.zeros(n)
    for h in range(horizon):
        Th = thetas[h]
        contrib = (Th @ sigma) ** 2  # (Th @ Sigma)_{ij}^2 per ogni i,j (colonna j = shock j)
        numer += contrib / sigma_diag[np.newaxis, :]
        denom += np.sum((Th @ sigma) * Th, axis=1)  # e_i' Th Sigma Th' e_i

    denom = np.where(denom <= 0, np.finfo(float).eps, denom)
    theta_raw = numer / denom[:, np.newaxis]  # riga i: contributo di ciascun j a i
    theta_norm = theta_raw / theta_raw.sum(axis=1, keepdims=True)

    spillover_df = pd.DataFrame(theta_norm, index=tickers, columns=tickers)

    # Total spillover index: media della quota "off-diagonal" (spillover da altri)
    off_diag_sum = theta_norm.sum() - np.trace(theta_norm)
    total_spillover_index = 100.0 * off_diag_sum / n

    G = nx.DiGraph()
    G.add_nodes_from(tickers)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i != j and theta_norm[i, j] > 1e-6:
                # arco j -> i: j contribuisce alla varianza dell'errore di i
                G.add_edge(tj, ti, weight=float(theta_norm[i, j]))

    logger.info(
        "Rete di Diebold-Yilmaz costruita: %d nodi, total spillover index = %.2f%%",
        n, total_spillover_index,
    )
    return G, spillover_df, float(total_spillover_index)


# --------------------------------------------------------------------------
# Rete economico-strutturale: classificazione settoriale
# --------------------------------------------------------------------------

def sector_network(tickers: List[str], sector_map: Dict[str, str]) -> nx.Graph:
    """
    Rete basata su classificazione industriale (GICS-like): un arco (non
    pesato, peso=1) tra ogni coppia di titoli appartenenti allo stesso settore.
    """
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i, j in itertools.combinations(tickers, 2):
        if sector_map.get(i) == sector_map.get(j):
            G.add_edge(i, j, weight=1.0)
    logger.info("Rete settoriale costruita: %d nodi, %d archi", G.number_of_nodes(), G.number_of_edges())
    return G


# --------------------------------------------------------------------------
# Utility comune ai modelli GNAR/GNARX/GNGARCH: matrici di vicinato stage-r
# --------------------------------------------------------------------------

def build_stage_weight_matrices(
    G: nx.Graph, node_order: List[str], max_stage: int, weighting: str = "equal"
) -> List[np.ndarray]:
    """
    Per un grafo G (diretto o meno, pesato o meno), calcola la lista
    [S_1, ..., S_max_stage] di matrici (N,N) riga-stocastiche tali che
    (S_r @ Y)[i] = media pesata dei valori dei vicini di ORDINE ESATTO r
    del nodo i (r-hop shortest path), coerente con la definizione
    N^(r)(i) e w_ij^(r) del modello GNAR.

    weighting:
      - 'equal'      : w_ij^(r) = 1 / |N^(r)(i)|  (pesatura uniforme, default
                        del pacchetto GNAR originale)
      - 'edge_weight': w_ij^(r) proporzionale al peso d'arco originale
                        (solo per r=1; per r>1 ricade su 'equal' poiché il
                        peso composito multi-hop non è definito nei dati)
    """
    nodes = node_order
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    is_directed = G.is_directed()

    matrices = [np.zeros((n, n)) for _ in range(max_stage)]

    for node in nodes:
        i = idx[node]
        if is_directed:
            # Per grafi diretti, i "vicini in ingresso" di i sono i j tali che
            # esiste un arco j->...->i: usiamo il grafo invertito per la
            # distanza (chi influenza i), coerente con l'uso in GNAR(X)
            # dove Y_{j,t-p} di un "predittore" j entra nell'equazione di i.
            lengths = nx.single_source_shortest_path_length(G.reverse(copy=False), node, cutoff=max_stage)
        else:
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=max_stage)

        by_stage: Dict[int, List[str]] = {}
        for other, d in lengths.items():
            if d == 0:
                continue
            by_stage.setdefault(d, []).append(other)

        for r in range(1, max_stage + 1):
            neighbors = by_stage.get(r, [])
            if not neighbors:
                continue
            if weighting == "edge_weight" and r == 1:
                if is_directed:
                    weights = np.array([G[other][node].get("weight", 1.0) for other in neighbors])
                else:
                    weights = np.array([G[node][other].get("weight", 1.0) for other in neighbors])
                weights = np.clip(weights, 1e-12, None)
                weights = weights / weights.sum()
            else:
                weights = np.full(len(neighbors), 1.0 / len(neighbors))

            for other, w in zip(neighbors, weights):
                matrices[r - 1][i, idx[other]] = w

    return matrices
