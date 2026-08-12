"""
src/models/gnar.py
====================
Implementazione del modello Generalised Network Autoregressive (GNAR) e
della sua estensione con covariate esogene (GNARX), secondo la
formalizzazione:

    Y_{i,t} = sum_{p=1}^P [ beta_{p,0} Y_{i,t-p}
                + sum_{r=1}^{R_p} beta_{p,r} sum_{j in N^(r)(i)} w_ij^(r) Y_{j,t-p} ]
              + sum_{k,q} gamma_{k,q} X_{k,t-q}
              + eps_{i,t}

Viene stimata la variante "globale" (coefficienti beta condivisi tra tutti
i nodi, coerente con la formulazione del documento di riferimento, dove
beta_{p,0} e beta_{p,r} non sono indicizzati da i). Questo riduce la
dimensionalità a sum_p(1+R_p) + (#esogene * #lag esogeni) parametri, contro
gli N^2*P di un VAR non vincolato.

La stima avviene impilando le osservazioni (nodo, tempo) in un unico
problema di regressione lineare (OLS), sfruttando le matrici di vicinato
S_r prodotte da `src/graphs/network_builders.build_stage_weight_matrices`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class GNARResult:
    coef_names: List[str]
    coefficients: np.ndarray
    intercept: float
    residuals: pd.DataFrame
    fitted: pd.DataFrame
    r_squared: float
    n_obs_used: int


class GNAR:
    """
    Modello GNAR(P, [R_1,...,R_P]) globale, con estensione opzionale GNARX
    (covariate esogene broadcast su tutti i nodi, es. tassi Fed/BCE).

    Parametri
    ---------
    P : ordine temporale massimo (numero di ritardi propri).
    R : lista di lunghezza P, R[p-1] = ordine massimo di vicinato al ritardo p.
    stage_matrices : lista di matrici (N,N) [S_1,...,S_Rmax] riga-stocastiche,
        prodotte da `build_stage_weight_matrices` sul grafo scelto.
    exog_lags : lista di lag (in periodi) da applicare a ciascuna colonna
        esogena (es. [1, 5] per tassi Fed/BCE ritardati di 1 e 5 giorni).
    fit_intercept : se True stima un intercetta globale condivisa.
    """

    def __init__(
        self,
        P: int,
        R: Sequence[int],
        stage_matrices: List[np.ndarray],
        exog_lags: Optional[Sequence[int]] = None,
        fit_intercept: bool = True,
    ):
        assert len(R) == P, "R deve avere esattamente P elementi (uno per ritardo)"
        self.P = P
        self.R = list(R)
        self.stage_matrices = stage_matrices
        self.exog_lags = list(exog_lags) if exog_lags else []
        self.fit_intercept = fit_intercept
        self._model: Optional[LinearRegression] = None
        self._feature_names: List[str] = []
        self._node_order: List[str] = []

    # ------------------------------------------------------------------
    def _build_design(
        self, Y: pd.DataFrame, exog: Optional[pd.DataFrame] = None
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """
        Costruisce il design matrix impilato (nodo,tempo) -> feature.
        Ritorna (X_long, y_long) entrambi con MultiIndex (time, node).
        """
        nodes = list(Y.columns)
        n = len(nodes)
        self._node_order = nodes

        max_r_needed = max(self.R) if self.R else 0
        assert len(self.stage_matrices) >= max_r_needed, (
            f"Servono almeno {max_r_needed} matrici di vicinato (stage_matrices), "
            f"ne sono state fornite {len(self.stage_matrices)}"
        )

        feature_cols: Dict[str, pd.DataFrame] = {}
        for p in range(1, self.P + 1):
            own_lag = Y.shift(p)
            feature_cols[f"own_lag{p}"] = own_lag
            for r in range(1, self.R[p - 1] + 1):
                S_r = self.stage_matrices[r - 1]
                net_lag = own_lag.values @ S_r.T  # (T, N): media pesata dei vicini di ordine r al ritardo p
                feature_cols[f"net_lag{p}_r{r}"] = pd.DataFrame(net_lag, index=Y.index, columns=Y.columns)

        exog_broadcast_cols: Dict[str, pd.DataFrame] = {}
        if exog is not None and self.exog_lags:
            for col in exog.columns:
                for q in self.exog_lags:
                    s = exog[col].shift(q)
                    # broadcast: stessa colonna macro replicata su tutti i nodi (coefficiente condiviso)
                    exog_broadcast_cols[f"{col}_lag{q}"] = pd.DataFrame(
                        np.tile(s.values.reshape(-1, 1), (1, n)), index=Y.index, columns=Y.columns
                    )

        self._feature_names = list(feature_cols.keys()) + list(exog_broadcast_cols.keys())
        all_cols = {**feature_cols, **exog_broadcast_cols}

        # Ogni DataFrame (T,N) viene "stackato" in una Series lunga (T*N,) con
        # MultiIndex (time, node); si concatenano tutte le feature per colonna.
        stacked_series = {name: df.stack() for name, df in all_cols.items()}
        X_long = pd.concat(stacked_series, axis=1)
        X_long.index.set_names(["__time__", "__node__"], inplace=True)

        y_long = Y.stack()
        y_long.index.set_names(["__time__", "__node__"], inplace=True)

        combined = X_long.copy()
        combined["__target__"] = y_long
        combined = combined.dropna()

        X_long = combined[self._feature_names]
        y_long = combined["__target__"]
        return X_long, y_long

    # ------------------------------------------------------------------
    def fit(self, Y: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> GNARResult:
        X_long, y_long = self._build_design(Y, exog)

        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(X_long.values, y_long.values)
        self._model = model

        y_hat_long = pd.Series(model.predict(X_long.values), index=X_long.index)
        resid_long = y_long - y_hat_long

        fitted = y_hat_long.unstack(level="__node__").reindex(columns=Y.columns)
        residuals = resid_long.unstack(level="__node__").reindex(columns=Y.columns)

        ss_res = float((resid_long ** 2).sum())
        ss_tot = float(((y_long - y_long.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return GNARResult(
            coef_names=self._feature_names,
            coefficients=model.coef_.copy(),
            intercept=float(model.intercept_) if self.fit_intercept else 0.0,
            residuals=residuals,
            fitted=fitted,
            r_squared=r2,
            n_obs_used=len(y_long),
        )

    # ------------------------------------------------------------------
    def predict_one_step(
        self, Y_history: pd.DataFrame, exog_history: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Prevede Y_{.,t+1} dati gli ultimi max(P) valori osservati per ciascun
        nodo (Y_history) e, se il modello è GNARX, gli ultimi valori delle
        esogene necessari ai lag configurati (exog_history).
        """
        assert self._model is not None, "Il modello deve essere stimato con .fit() prima di prevedere"
        nodes = list(Y_history.columns)

        feats: Dict[str, np.ndarray] = {}
        for p in range(1, self.P + 1):
            own_lag_row = Y_history.iloc[-p].values  # valore a t+1-p
            feats[f"own_lag{p}"] = own_lag_row
            for r in range(1, self.R[p - 1] + 1):
                S_r = self.stage_matrices[r - 1]
                feats[f"net_lag{p}_r{r}"] = S_r @ own_lag_row

        exog_feats: Dict[str, float] = {}
        if exog_history is not None and self.exog_lags:
            for col in exog_history.columns:
                for q in self.exog_lags:
                    exog_feats[f"{col}_lag{q}"] = exog_history[col].iloc[-q]

        X_row = []
        for name in self._feature_names:
            if name in feats:
                X_row.append(feats[name])
            else:
                X_row.append(np.full(len(nodes), exog_feats[name]))
        X_row = np.column_stack(X_row)  # (N, n_features)

        y_pred = self._model.predict(X_row)
        return pd.Series(y_pred, index=nodes)


def build_gnar_stage_matrices(graph, node_order: List[str], max_stage: int, weighting: str = "equal"):
    """Shortcut che richiama `network_builders.build_stage_weight_matrices` (import lazy per evitare cicli)."""
    from src.graphs.network_builders import build_stage_weight_matrices

    return build_stage_weight_matrices(graph, node_order, max_stage, weighting)
