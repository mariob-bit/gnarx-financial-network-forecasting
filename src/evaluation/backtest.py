"""
src/evaluation/backtest.py
============================
Motore di backtest rolling (walk-forward) "out-of-sample" per la previsione
di rendimenti/varianza: ad ogni passo si usa solo l'informazione disponibile
fino a t (nessun lookahead), si ristima periodicamente il grafo e i modelli
(costoso, quindi controllato da `refit_every`), e si prevede un passo avanti.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    mean_forecasts: pd.DataFrame
    variance_forecasts: pd.DataFrame
    actual_returns: pd.DataFrame
    n_refits: int


def rolling_backtest(
    returns: pd.DataFrame,
    exog: pd.DataFrame,
    build_and_fit_fn: Callable[[pd.DataFrame, pd.DataFrame], object],
    predict_fn: Callable[[object, pd.DataFrame, pd.DataFrame], "tuple[pd.Series, pd.Series]"],
    train_window: int,
    test_horizon: int,
    refit_every: int = 10,
    max_lag_needed: int = 5,
) -> BacktestResult:
    """
    Esegue un backtest rolling generico e indipendente dal tipo di modello:

    build_and_fit_fn(Y_train, exog_train) -> modello stimato
    predict_fn(modello, Y_history, exog_history) -> (mean_pred, var_pred)

    Il chiamante (script di orchestrazione) fornisce le due funzioni per
    incapsulare la logica specifica del modello (GNARX+DST-ARCH, GNARX+GNGARCH,
    oppure benchmark AR+GARCH). In particolare, `predict_fn` è responsabile
    di recuperare l'ultimo residuo noto (eps_last) dallo stato interno del
    modello stimato (es. dai residui del mean_model), cosicché il motore di
    backtest resti identico e agnostico rispetto al tipo di modello.

    Parametri
    ---------
    train_window : dimensione della finestra di stima (in osservazioni).
    test_horizon : numero di passi out-of-sample da prevedere.
    refit_every  : ogni quanti passi ri-stimare il modello (costo computazionale).
    max_lag_needed : massimo lag richiesto dal modello per generare le feature
        (serve per determinare quante osservazioni di history servono a predict_fn).
    """
    n_total = len(returns)
    start_idx = n_total - test_horizon
    assert start_idx - train_window >= max_lag_needed, "Serie troppo corta per la configurazione di backtest richiesta"

    mean_preds = []
    var_preds = []
    actuals = []
    model = None
    n_refits = 0

    for step, t_idx in enumerate(tqdm(range(start_idx, n_total), desc="Rolling backtest")):
        train_start = t_idx - train_window
        Y_train = returns.iloc[train_start:t_idx]
        exog_train = exog.iloc[train_start:t_idx]

        if model is None or step % refit_every == 0:
            model = build_and_fit_fn(Y_train, exog_train)
            n_refits += 1

        Y_history = returns.iloc[:t_idx]
        exog_history = exog.iloc[:t_idx]

        mean_pred, var_pred = predict_fn(model, Y_history, exog_history)

        mean_preds.append(mean_pred.rename(returns.index[t_idx]))
        var_preds.append(var_pred.rename(returns.index[t_idx]))
        actuals.append(returns.iloc[t_idx])

    mean_forecasts = pd.DataFrame(mean_preds)
    variance_forecasts = pd.DataFrame(var_preds)
    actual_returns = pd.DataFrame(actuals)
    actual_returns.index = mean_forecasts.index

    logger.info("Backtest completato: %d passi, %d refit del modello", test_horizon, n_refits)
    return BacktestResult(
        mean_forecasts=mean_forecasts,
        variance_forecasts=variance_forecasts,
        actual_returns=actual_returns,
        n_refits=n_refits,
    )
