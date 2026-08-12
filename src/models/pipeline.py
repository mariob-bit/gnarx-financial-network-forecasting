"""
src/models/pipeline.py
========================
Modello finale della pipeline: GNARX sulla media + DST-ARCH sulla varianza.

Fasi:
  1. Si stima GNARX(P, [R_p]) con esogene (Fed funds rate, tasso BCE) sui
     rendimenti -> media condizionata mu_{i,t} e residui eps_{i,t}.
  2. Si stima DST-ARCH sui residui eps_{i,t} -> varianza condizionata h_{i,t}.
  3. La previsione one-step-ahead combina i due componenti: fornisce sia il
     rendimento atteso sia la varianza attesa (utile per intervalli di
     previsione, VaR parametrico, ecc.).

Include anche `GNGARCHMeanVariancePipeline` come alternativa con GNGARCH al
posto di DST-ARCH, usata come modello di confronto nel backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .gnar import GNAR
from .variance_models import DSTARCH, GNGARCH


@dataclass
class StepForecast:
    mean: pd.Series
    variance: pd.Series


class GNARXDSTARCHPipeline:
    """Modello finale: GNARX (media) + DST-ARCH (varianza)."""

    def __init__(
        self,
        gnar_P: int,
        gnar_R: List[int],
        stage_matrices_mean: List[np.ndarray],
        exog_lags: List[int],
        dst_ar_order: int = 1,
        dst_log_shift_c: float = 1e-6,
        W_variance: Optional[np.ndarray] = None,
    ):
        self.mean_model = GNAR(P=gnar_P, R=gnar_R, stage_matrices=stage_matrices_mean, exog_lags=exog_lags)
        self.variance_model = DSTARCH(ar_order=dst_ar_order, log_shift_c=dst_log_shift_c)
        self._W_variance = W_variance
        self._gnar_P = gnar_P
        self._fitted = False
        self._last_eps: Optional[np.ndarray] = None

    def fit(self, Y: pd.DataFrame, exog: pd.DataFrame, W_variance: Optional[np.ndarray] = None):
        mean_res = self.mean_model.fit(Y, exog=exog)
        residuals = mean_res.residuals.dropna(how="any")
        W = W_variance if W_variance is not None else self._W_variance
        assert W is not None, "Serve la matrice di adiacenza W per il modello di varianza"
        var_res = self.variance_model.fit(residuals, W)
        self._last_eps = residuals.iloc[-1].values
        self._fitted = True
        return mean_res, var_res

    def predict_one_step(self, Y_history: pd.DataFrame, exog_history: pd.DataFrame) -> StepForecast:
        assert self._fitted, "Il modello deve essere stimato con .fit() prima di prevedere"
        mean_pred = self.mean_model.predict_one_step(Y_history, exog_history)
        var_pred = self.variance_model.forecast_one_step(self._last_eps)
        return StepForecast(mean=mean_pred, variance=pd.Series(var_pred, index=mean_pred.index))


class GNARGNGARCHPipeline:
    """Modello di confronto: GNARX (media) + GNGARCH (varianza)."""

    def __init__(
        self,
        gnar_P: int,
        gnar_R: List[int],
        stage_matrices_mean: List[np.ndarray],
        exog_lags: List[int],
        gngarch_R: int,
        stage_matrices_var: List[np.ndarray],
        distribution: str = "t",
        threshold: bool = False,
    ):
        self.mean_model = GNAR(P=gnar_P, R=gnar_R, stage_matrices=stage_matrices_mean, exog_lags=exog_lags)
        self.variance_model = GNGARCH(R=gngarch_R, distribution=distribution, threshold=threshold)
        self._stage_matrices_var = stage_matrices_var
        self._fitted = False
        self._last_h: Optional[np.ndarray] = None
        self._last_eps: Optional[np.ndarray] = None

    def fit(self, Y: pd.DataFrame, exog: pd.DataFrame):
        mean_res = self.mean_model.fit(Y, exog=exog)
        residuals = mean_res.residuals.dropna(how="any")
        var_res = self.variance_model.fit(residuals, self._stage_matrices_var)
        self._last_h = var_res.conditional_variance.iloc[-1].values
        self._last_eps = residuals.iloc[-1].values
        self._fitted = True
        return mean_res, var_res

    def predict_one_step(self, Y_history: pd.DataFrame, exog_history: pd.DataFrame) -> StepForecast:
        assert self._fitted
        mean_pred = self.mean_model.predict_one_step(Y_history, exog_history)
        var_pred = self.variance_model.forecast_one_step(self._last_eps, self._last_h)
        self._last_h = var_pred
        return StepForecast(mean=mean_pred, variance=pd.Series(var_pred, index=mean_pred.index))
