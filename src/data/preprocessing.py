"""
src/data/preprocessing.py
==========================
Allineamento delle serie storiche (prezzi + tassi Fed/BCE), calcolo dei
rendimenti logaritmici, e diagnostica preliminare (ACF/PACF, test di
stazionarietà ADF, test di autocorrelazione Ljung-Box).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti logaritmici: r_t = ln(P_t) - ln(P_{t-1})."""
    log_p = np.log(prices.astype(float))
    returns = log_p.diff().dropna(how="all")
    return returns


def align_all_series(
    prices: pd.DataFrame,
    fed: pd.Series,
    ecb: pd.Series,
    min_valid_fraction: float = 0.90,
) -> Dict[str, pd.DataFrame]:
    """
    Allinea prezzi e tassi su un indice comune di giorni lavorativi (business days),
    gestendo festività non coincidenti tra borse/paesi con forward-fill (i tassi e
    i prezzi non aggiornati restano validi fino al valore successivo, come è
    finanziariamente corretto: es. festività USA con BCE aperta o viceversa).

    Rimuove i titoli con più di `1 - min_valid_fraction` di dati mancanti nel
    periodo comune, per evitare che una singola serie "corta" (es. IPO recente)
    riduca eccessivamente il campione.
    """
    common_index = prices.index.union(fed.index).union(ecb.index)
    common_index = pd.bdate_range(common_index.min(), common_index.max())

    prices_aligned = prices.reindex(common_index).ffill()
    fed_aligned = fed.reindex(common_index).ffill().bfill()
    ecb_aligned = ecb.reindex(common_index).ffill().bfill()

    valid_frac = prices_aligned.notna().mean()
    dropped = valid_frac[valid_frac < min_valid_fraction].index.tolist()
    if dropped:
        logger.warning("Rimossi %d ticker con troppi dati mancanti: %s", len(dropped), dropped)
        prices_aligned = prices_aligned.drop(columns=dropped)

    # Dopo l'allineamento iniziale, elimina le righe iniziali ancora tutte-NaN
    # (es. periodo precedente alla quotazione di tutti i titoli)
    prices_aligned = prices_aligned.dropna(how="any")
    common_valid_index = prices_aligned.index
    fed_aligned = fed_aligned.reindex(common_valid_index).ffill().bfill()
    ecb_aligned = ecb_aligned.reindex(common_valid_index).ffill().bfill()

    returns = compute_log_returns(prices_aligned)
    fed_aligned = fed_aligned.reindex(returns.index)
    ecb_aligned = ecb_aligned.reindex(returns.index)

    logger.info(
        "Allineamento completato: %d osservazioni, %d asset, periodo %s -> %s",
        len(returns), returns.shape[1], returns.index.min().date(), returns.index.max().date(),
    )

    return {
        "prices": prices_aligned,
        "returns": returns,
        "fed": fed_aligned,
        "ecb": ecb_aligned,
    }


@dataclass
class SeriesDiagnostics:
    ticker: str
    adf_stat: float
    adf_pvalue: float
    is_stationary_5pct: bool
    ljung_box_stat: float
    ljung_box_pvalue: float
    has_autocorrelation_5pct: bool
    acf_values: np.ndarray
    pacf_values: np.ndarray


def diagnose_series(returns: pd.DataFrame, n_lags: int = 20, lb_lags: int = 10) -> pd.DataFrame:
    """
    Calcola, per ciascuna serie di rendimenti:
      - ADF test (stazionarietà)
      - Ljung-Box test (autocorrelazione residua fino a `lb_lags` ritardi)
      - ACF e PACF fino a `n_lags` ritardi

    Ritorna un DataFrame riassuntivo (una riga per asset) con le colonne
    principali; ACF/PACF completi sono disponibili tramite `diagnose_series_full`.
    """
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        lb = acorr_ljungbox(s, lags=[lb_lags], return_df=True)
        rows.append(
            {
                "ticker": col,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "stazionaria_5pct": adf_p < 0.05,
                "ljung_box_stat": lb["lb_stat"].iloc[0],
                "ljung_box_pvalue": lb["lb_pvalue"].iloc[0],
                "autocorrelata_5pct": lb["lb_pvalue"].iloc[0] < 0.05,
                "media": s.mean(),
                "std": s.std(),
                "skew": s.skew(),
                "kurtosi": s.kurtosis(),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def compute_acf_pacf(returns: pd.DataFrame, n_lags: int = 20) -> Dict[str, Dict[str, np.ndarray]]:
    """Ritorna {ticker: {'acf': array, 'pacf': array}} per plotting/ispezione."""
    out = {}
    for col in returns.columns:
        s = returns[col].dropna()
        out[col] = {
            "acf": acf(s, nlags=n_lags, fft=True),
            "pacf": pacf(s, nlags=n_lags),
        }
    return out


def cross_correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Matrice di correlazione (contemporanea) tra i rendimenti degli asset."""
    return returns.corr(method=method)
