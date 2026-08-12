"""
src/data/synthetic.py
======================
Genera un dataset sintetico con la STESSA interfaccia di `download.download_all`
(prices, fed, ecb), ma prodotto localmente da un processo generatore noto
(GNAR(1,[1]) sulla media + GNGARCH(1,1,[1],[1]) sulla varianza, su un grafo
a struttura di comunità coerente con i settori di `config.yaml`).

Serve esclusivamente a:
  1. Validare che l'intera pipeline (preprocessing -> grafi -> modelli ->
     backtest -> metriche) sia eseguibile end-to-end senza errori.
  2. Fornire un run dimostrativo quando l'ambiente non ha accesso a Internet
     (es. Yahoo Finance/FRED/ECB non raggiungibili).

Non è pensato per validare empiricamente la bontà dei modelli sui mercati
reali: quello richiede dati reali, ottenibili con `src/data/download.py`.
"""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .download import flatten_universe


def build_true_community_graph(universe: Dict[str, list], seed: int = 42, p_in: float = 0.55, p_out: float = 0.04) -> nx.Graph:
    """
    Costruisce un grafo "vero" a struttura di comunità: nodi dello stesso
    settore hanno probabilità p_in di essere connessi, nodi di settori
    diversi probabilità p_out. Usato come DGP di riferimento per i dati
    sintetici (e come termine di paragone per i grafi stimati dai dati).
    """
    rng = np.random.default_rng(seed)
    tickers = flatten_universe(universe)
    G = nx.Graph()
    G.add_nodes_from(tickers)

    ticker_to_sector = {t: s for s, ts in universe.items() for t in ts}

    for i, ti in enumerate(tickers):
        for tj in tickers[i + 1:]:
            same_sector = ticker_to_sector[ti] == ticker_to_sector[tj]
            p = p_in if same_sector else p_out
            if rng.random() < p:
                w = rng.uniform(0.3, 1.0)
                G.add_edge(ti, tj, weight=w)

    # Garantisce connessione (nessun nodo isolato): collega eventuali nodi
    # isolati al nodo più "centrale" del proprio settore.
    for t in tickers:
        if G.degree(t) == 0:
            sector = ticker_to_sector[t]
            candidates = [x for x in universe[sector] if x != t]
            if candidates:
                other = rng.choice(candidates)
                G.add_edge(t, other, weight=rng.uniform(0.3, 1.0))
    return G


def _row_normalized_adjacency(G: nx.Graph, node_order: list) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=node_order, weight="weight")
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def _simulate_synthetic_rate(n_obs: int, seed: int, level0: float, vol: float, lo: float, hi: float) -> np.ndarray:
    """Simula una serie di tassi 'a scalini' (cambia raramente), plausibile per Fed/BCE."""
    rng = np.random.default_rng(seed)
    n_changes = max(3, n_obs // 90)  # un cambio di tasso circa ogni ~90 osservazioni
    change_points = np.sort(rng.choice(np.arange(5, n_obs), size=n_changes, replace=False))
    levels = [level0]
    for _ in range(n_changes):
        step = rng.choice([-0.25, -0.5, 0.0, 0.25, 0.5], p=[0.15, 0.05, 0.35, 0.35, 0.10])
        levels.append(float(np.clip(levels[-1] + step, lo, hi)))
    series = np.full(n_obs, levels[0])
    for cp, lvl in zip(change_points, levels[1:]):
        series[cp:] = lvl
    series = series + rng.normal(0, vol, n_obs)  # piccolo rumore di misura
    return np.clip(series, lo, hi)


def generate_synthetic_dataset(config: dict) -> Tuple[Dict[str, object], nx.Graph]:
    """
    Genera prezzi sintetici (ricostruiti da rendimenti simulati via un vero
    DGP GNAR+GNGARCH su grafo), più serie sintetiche di tassi Fed/BCE.

    Ritorna (data_dict, true_graph) dove data_dict ha chiavi 'prices','fed','ecb'
    con la stessa struttura di `download.download_all`.
    """
    data_cfg = config["data"]
    sim_cfg = config["synthetic_demo"]
    universe = data_cfg["universe"]
    tickers = flatten_universe(universe)
    n = len(tickers)
    T = int(sim_cfg["n_obs"])
    seed = int(sim_cfg["seed"])
    rng = np.random.default_rng(seed)

    true_graph = build_true_community_graph(universe, seed=seed)
    S1 = _row_normalized_adjacency(true_graph, tickers)  # (N,N) media pesata vicini stage-1

    # --- Parametri del vero DGP (scelti per stazionarietà e realismo) ---
    beta0 = 0.03      # autoregressione diretta debole (rendimenti ~ rumore + rete)
    beta_net = 0.10    # effetto di rete sulla media (contagio informativo)
    mu = rng.normal(0.0003, 0.0001, n)  # piccolo drift positivo eterogeneo (stile equity)

    omega = rng.uniform(1e-6, 4e-6, n)
    alpha = 0.06
    alpha_net = 0.05
    beta_g = 0.85
    beta_g_net = 0.02
    # stazionarietà (in media, dato che S1 è riga-stocastica -> autovalore dominante 1):
    assert alpha + alpha_net + beta_g + beta_g_net < 1.0, "DGP di volatilità non stazionario"

    eps = np.zeros((T, n))
    h = np.zeros((T, n))
    Y = np.zeros((T, n))
    h[0] = omega / (1 - alpha - alpha_net - beta_g - beta_g_net)

    # Finestra di "stress" sintetica (es. ultimo ~15% del campione): un
    # fattore comune con loading elevato spinge la correlazione media tra
    # gli asset molto in alto, simulando un rally/rottura sincronizzati.
    # Serve a validare che l'indicatore di regime spettrale (lambda1_share,
    # HMM) rilevi davvero l'evento nel run dimostrativo.
    stress_start = int(T * 0.80)
    stress_end = int(T * 0.90)
    common_loading = np.zeros(T)
    common_loading[stress_start:stress_end] = 0.85  # loading elevato = alta correlazione media
    common_shock = rng.standard_normal(T) * 0.012

    for t in range(1, T):
        net_eps2_lag = S1 @ (eps[t - 1] ** 2)
        net_h_lag = S1 @ h[t - 1]
        h[t] = omega + alpha * eps[t - 1] ** 2 + alpha_net * net_eps2_lag + beta_g * h[t - 1] + beta_g_net * net_h_lag
        h[t] = np.clip(h[t], 1e-10, None)

        z = rng.standard_t(df=7, size=n) * np.sqrt(5 / 7)  # innovazioni t-Student standardizzate
        idio_shock = np.sqrt(h[t]) * z
        # combina lo shock idiosincratico/di rete con il fattore comune (loading
        # costante fuori dalla finestra di stress -> quasi zero, quindi il DGP
        # base resta quello GNAR+GNGARCH descritto sopra)
        eps[t] = np.sqrt(1 - common_loading[t] ** 2) * idio_shock + common_loading[t] * common_shock[t]

        net_Y_lag = S1 @ Y[t - 1]
        Y[t] = mu + beta0 * Y[t - 1] + beta_net * net_Y_lag + eps[t]

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)
    returns_true = pd.DataFrame(Y, index=dates, columns=tickers)

    # Ricostruisce prezzi sintetici plausibili (prezzo iniziale casuale 50-500) via cumprod
    p0 = rng.uniform(50, 500, n)
    prices = p0 * np.exp(returns_true.cumsum())
    prices = prices.round(4)

    fed_vals = _simulate_synthetic_rate(T, seed=seed + 1, level0=0.25, vol=0.02, lo=0.0, hi=5.5)
    ecb_vals = _simulate_synthetic_rate(T, seed=seed + 2, level0=0.0, vol=0.02, lo=0.0, hi=4.5)
    fed = pd.Series(fed_vals, index=dates, name="fed_funds_rate")
    ecb = pd.Series(ecb_vals, index=dates, name="ecb_main_refi_rate")

    data = {"prices": prices, "fed": fed, "ecb": ecb}
    return data, true_graph
