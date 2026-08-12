"""
src/models/benchmarks.py
==========================
Modelli benchmark "non di rete", stimati indipendentemente per ciascun
asset, usati come termine di paragone per quantificare il valore aggiunto
delle componenti di rete (GNARX vs AR; GNGARCH/DST-ARCH vs GARCH(1,1)
univariato), in linea con l'evidenza empirica citata nel documento di
riferimento (i modelli spazio-temporali su grafo riducono l'RMSFE rispetto
ai benchmark univariati/basati su correlazioni lineari).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class ARBenchmark:
    """AR(p) indipendente per ciascun asset (nessuna informazione di rete)."""

    def __init__(self, lags: int = 2):
        self.lags = lags
        self._fitted: Dict[str, "AutoRegResultsWrapper"] = {}

    def fit(self, Y: pd.DataFrame) -> "ARBenchmark":
        self._fitted = {}
        for col in Y.columns:
            s = Y[col].dropna()
            model = AutoReg(s, lags=self.lags, old_names=False)
            self._fitted[col] = model.fit()
        return self

    def predict_one_step(self, Y_history: pd.DataFrame) -> pd.Series:
        preds = {}
        for col in Y_history.columns:
            res = self._fitted[col]
            params = res.params.values  # [const, lag1, ..., lagp]
            hist = Y_history[col].values[-self.lags:][::-1]  # più recente prima
            pred = params[0] + np.dot(params[1:], hist)
            preds[col] = pred
        return pd.Series(preds)


class GARCH11Benchmark:
    """GARCH(1,1) indipendente per ciascun asset (nessuna informazione di rete), via `arch`."""

    def __init__(self, distribution: str = "t"):
        self.distribution = distribution
        self._fitted = {}

    def fit(self, eps: pd.DataFrame) -> "GARCH11Benchmark":
        from arch import arch_model

        self._fitted = {}
        for col in eps.columns:
            s = eps[col].dropna() * 100.0  # scaling numerico standard per `arch`
            dist = "t" if self.distribution == "t" else "normal"
            am = arch_model(s, mean="Zero", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
            res = am.fit(disp="off")
            self._fitted[col] = res
        return self

    def conditional_variance(self) -> pd.DataFrame:
        out = {}
        for col, res in self._fitted.items():
            out[col] = res.conditional_volatility ** 2 / 100.0 ** 2
        return pd.DataFrame(out)

    def forecast_one_step(self) -> pd.Series:
        preds = {}
        for col, res in self._fitted.items():
            f = res.forecast(horizon=1, reindex=False)
            preds[col] = f.variance.values[-1, 0] / 100.0 ** 2
        return pd.Series(preds)
