"""
src/utils/plotting.py
=======================
Funzioni di visualizzazione per ispezione dei grafi, diagnostica ACF/PACF,
e confronto previsioni rolling vs valori realizzati (media e volatilità).
Tutte le funzioni salvano il grafico su file (nessun plt.show(), adatto a
esecuzione headless / CI).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_graph(G: nx.Graph, title: str, path: str, sector_map: Optional[Dict[str, str]] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=42, weight="weight" if nx.get_edge_attributes(G, "weight") else None)

    if sector_map:
        sectors = sorted(set(sector_map.values()))
        cmap = plt.get_cmap("tab10")
        color_map = {s: cmap(i % 10) for i, s in enumerate(sectors)}
        node_colors = [color_map.get(sector_map.get(n, ""), "grey") for n in G.nodes()]
    else:
        node_colors = "#4C72B0"

    directed = G.is_directed()
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=550, ax=ax, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=directed, alpha=0.5, width=1.0, arrowsize=10)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_acf_pacf_grid(acf_pacf: Dict[str, Dict[str, np.ndarray]], path: str, max_assets: int = 6) -> None:
    tickers = list(acf_pacf.keys())[:max_assets]
    fig, axes = plt.subplots(len(tickers), 2, figsize=(10, 2.2 * len(tickers)))
    if len(tickers) == 1:
        axes = axes.reshape(1, 2)
    for i, t in enumerate(tickers):
        acf_vals = acf_pacf[t]["acf"]
        pacf_vals = acf_pacf[t]["pacf"]
        axes[i, 0].stem(range(len(acf_vals)), acf_vals)
        axes[i, 0].set_title(f"ACF - {t}", fontsize=9)
        axes[i, 1].stem(range(len(pacf_vals)), pacf_vals)
        axes[i, 1].set_title(f"PACF - {t}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rolling_mean_forecast(actual: pd.DataFrame, forecast: pd.DataFrame, ticker: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual.index, actual[ticker], label="Rendimento realizzato", color="black", linewidth=1)
    ax.plot(forecast.index, forecast[ticker], label="Previsione GNARX (rolling)", color="crimson", linewidth=1, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"Previsione rolling della media (rendimenti) - {ticker}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rolling_volatility_forecast(actual_sq_ret: pd.DataFrame, forecast_var: pd.DataFrame, ticker: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_sq_ret.index, np.sqrt(actual_sq_ret[ticker]), label="Volatilità realizzata (|r_t| proxy)", color="black", linewidth=1)
    ax.plot(forecast_var.index, np.sqrt(forecast_var[ticker].clip(lower=0)), label="Volatilità prevista DST-ARCH (rolling)", color="crimson", linewidth=1, alpha=0.8)
    ax.set_title(f"Previsione rolling della volatilità - {ticker}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, column: str, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = metrics_df.drop(index="__ALL__", errors="ignore")[column].sort_values()
    ax.barh(data.index, data.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spectral_regime(features: pd.DataFrame, path: str) -> None:
    """Grafico a due pannelli: lambda1_share (+ n. autovalori significativi MP)
    e probabilità filtrata di regime di stress (HMM), con evidenziazione delle
    zone ad alta probabilità di stress."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(features.index, features["lambda1_share"], color="#4C72B0", linewidth=1.2,
                 label=r"$\lambda_1 / \sum \lambda_i$ (quota di varianza primo autovalore)")
    axes[0].set_title("Indicatore di regime spettrale: concentrazione della varianza (lambda1_share)")
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("quota")

    if "regime_prob_stress" in features.columns:
        axes[1].plot(features.index, features["regime_prob_stress"], color="crimson", linewidth=1.2,
                     label="P(regime di stress) - filtrata (HMM, forward-only)")
        axes[1].fill_between(features.index, 0, features["regime_prob_stress"], color="crimson", alpha=0.15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("probabilità")
    axes[1].legend(loc="upper left")
    axes[1].set_title("Probabilità di regime di stress (HMM Gaussiano a 2 stati, filtraggio causale)")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
