"""
src/evaluation/metrics.py
===========================
Metriche di accuratezza per le previsioni rolling out-of-sample, separate
per componente di media e di varianza, più un test di copertura del
Value-at-Risk come diagnostica congiunta media+varianza.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2


# --------------------------------------------------------------------------
# Metriche sulla media (rendimenti)
# --------------------------------------------------------------------------

def mean_forecast_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Metriche di accuratezza della previsione di media, per asset e in
    aggregato ('__ALL__'): RMSE, MAE, R2 out-of-sample, Directional Accuracy
    (quota di segno correttamente previsto).
    """
    common_idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]

    rows = {}
    for col in y_true.columns:
        yt = y_true[col].dropna()
        yp = y_pred[col].reindex(yt.index).dropna()
        yt = yt.reindex(yp.index)
        err = yt - yp
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        dir_acc = float(np.mean(np.sign(yt) == np.sign(yp))) if len(yt) > 0 else np.nan
        rows[col] = {"RMSE": rmse, "MAE": mae, "R2_oos": r2, "Directional_Accuracy": dir_acc, "n_obs": len(yt)}

    df = pd.DataFrame(rows).T
    # riga aggregata: pooling di tutti gli errori (non media delle metriche per asset)
    err_all = (y_true - y_pred).values.flatten()
    err_all = err_all[~np.isnan(err_all)]
    yt_all = y_true.values.flatten()
    yt_all = yt_all[~np.isnan(yt_all)]
    yp_all = y_pred.reindex(y_true.index).values.flatten()
    yp_all = yp_all[~np.isnan(yp_all)]
    dir_acc_all = float(np.mean(np.sign(yt_all) == np.sign(yp_all))) if len(yt_all) == len(yp_all) and len(yt_all) > 0 else np.nan
    df.loc["__ALL__"] = {
        "RMSE": float(np.sqrt(np.mean(err_all ** 2))),
        "MAE": float(np.mean(np.abs(err_all))),
        "R2_oos": np.nan,
        "Directional_Accuracy": dir_acc_all,
        "n_obs": len(err_all),
    }
    return df


# --------------------------------------------------------------------------
# Metriche sulla varianza (volatilità)
# --------------------------------------------------------------------------

def qlike_loss(realized_proxy: np.ndarray, forecast_variance: np.ndarray) -> float:
    """
    QLIKE: loss standard per la valutazione di previsioni di varianza,
    robusta al rumore della proxy della varianza realizzata (qui: rendimento
    al quadrato, in assenza di dati infragiornalieri):
        QLIKE = mean( log(h_hat) + realized / h_hat )
    """
    h = np.maximum(forecast_variance, 1e-12)
    return float(np.mean(np.log(h) + realized_proxy / h))


def variance_forecast_metrics(realized_sq_returns: pd.DataFrame, variance_forecast: pd.DataFrame) -> pd.DataFrame:
    """Metriche di accuratezza della previsione di varianza, per asset e in aggregato."""
    common_idx = realized_sq_returns.index.intersection(variance_forecast.index)
    r2 = realized_sq_returns.loc[common_idx]
    h = variance_forecast.loc[common_idx]

    rows = {}
    for col in r2.columns:
        rt = r2[col].dropna()
        ht = h[col].reindex(rt.index).dropna()
        rt = rt.reindex(ht.index)
        mse = float(np.mean((rt - ht) ** 2))
        qlike = qlike_loss(rt.values, ht.values)
        rows[col] = {"MSE_variance": mse, "QLIKE": qlike, "n_obs": len(rt)}

    df = pd.DataFrame(rows).T
    rt_all = r2.values.flatten()
    ht_all = h.reindex(r2.index).values.flatten()
    mask = ~np.isnan(rt_all) & ~np.isnan(ht_all)
    df.loc["__ALL__"] = {
        "MSE_variance": float(np.mean((rt_all[mask] - ht_all[mask]) ** 2)),
        "QLIKE": qlike_loss(rt_all[mask], ht_all[mask]),
        "n_obs": int(mask.sum()),
    }
    return df


def var_coverage_test(returns: pd.DataFrame, mean_forecast: pd.DataFrame, variance_forecast: pd.DataFrame, alpha: float = 0.05, dist_quantile: float = 1.645) -> pd.DataFrame:
    """
    Test di copertura del Value-at-Risk (Kupiec, 1995) per il livello
    (1-alpha): confronta la quota empirica di violazioni (rendimento < VaR)
    con quella attesa `alpha`, con test statistico Likelihood Ratio.

    VaR_{i,t} = mu_{i,t} - z_alpha * sqrt(h_{i,t})  (approssimazione parametrica).
    """
    common_idx = returns.index.intersection(mean_forecast.index).intersection(variance_forecast.index)
    r = returns.loc[common_idx]
    mu = mean_forecast.loc[common_idx]
    h = variance_forecast.loc[common_idx]

    rows = {}
    for col in r.columns:
        rt = r[col].dropna()
        idx = rt.index.intersection(mu[col].dropna().index).intersection(h[col].dropna().index)
        rt = rt.loc[idx]
        var_est = mu.loc[idx, col] - dist_quantile * np.sqrt(h.loc[idx, col])
        violations = (rt < var_est).astype(int)
        n = len(violations)
        x = int(violations.sum())
        pi_hat = x / n if n > 0 else np.nan

        if 0 < x < n:
            ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
            ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
            lr_stat = -2 * (ll_null - ll_alt)
            p_value = 1 - chi2.cdf(lr_stat, df=1)
        else:
            lr_stat, p_value = np.nan, np.nan

        rows[col] = {
            "n_obs": n, "n_violazioni": x, "quota_violazioni": pi_hat,
            "quota_attesa": alpha, "kupiec_LR_stat": lr_stat, "kupiec_pvalue": p_value,
        }
    return pd.DataFrame(rows).T
