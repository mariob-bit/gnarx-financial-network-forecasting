"""
src/graphs/spectral_regime.py
===============================
Indicatore di regime di mercato basato sullo spettro (autovalori) della
matrice di correlazione tra i rendimenti, calcolato su finestra mobile.

Razionale: in periodi normali gli autovalori della matrice di correlazione
sono dispersi (struttura settoriale riconoscibile). Nei momenti di rally o
rottura sincronizzati (crash, panic selling, squeeze), la correlazione media
tra i titoli sale bruscamente e il PRIMO autovalore (lambda_1) assorbe una
quota molto maggiore della varianza totale ("un solo fattore di mercato
domina tutto"). Questo modulo produce quattro feature, pensate per essere
usate come variabili ESOGENE aggiuntive nel modello GNARX:

  1. lambda1_share(t)      : lambda_1 / sum(lambda_i) sulla finestra [t-W+1, t]
  2. lambda1_share_roc(t)  : variazione di lambda1_share su `roc_lag` periodi
                              (segnale di "early warning": la velocità con cui
                              cambia anticipa spesso il prezzo).
  3. n_significant_mp(t)   : numero di autovalori che eccedono il bordo
                              superiore di Marchenko-Pastur (filtro RMT:
                              separa "segnale" da "rumore" nella matrice di
                              correlazione).
  4. regime_prob_stress(t) : probabilità FILTRATA (forward-only, quindi
                              causale: usa solo osservazioni fino a t) di
                              trovarsi nello stato "stress" di un Hidden
                              Markov Model Gaussiano a 2 stati stimato su
                              lambda1_share.

Tutte le feature sono costruite su finestre STRETTAMENTE PASSATE (nessun
lookahead): lambda1_share(t) usa solo rendimenti fino a t. Per l'HMM: i
PARAMETRI (transizioni, medie, varianze) vengono stimati una volta per
Expectation-Maximization sull'intera serie storica disponibile (prassi
comune, analoga alla calibrazione di un GARCH); le PROBABILITÀ DI STATO per
ciascun periodo t sono però ottenute per filtraggio (forward pass, non
smoothing) usando solo le osservazioni fino a t, per evitare che l'inferenza
del regime a un dato istante usi informazione futura.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def marchenko_pastur_bounds(n_assets: int, n_obs: int) -> "tuple[float, float]":
    """Bordi (lambda_-, lambda_+) della distribuzione di Marchenko-Pastur per
    una matrice di correlazione N x N stimata da T osservazioni iid (T>N)."""
    q = n_assets / n_obs
    lambda_plus = (1 + np.sqrt(q)) ** 2
    lambda_minus = (1 - np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus


def rolling_eigen_spectrum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calcola, per ogni t >= window, lo spettro della matrice di correlazione
    sulla finestra mobile [t-window+1, t] e ne estrae lambda1_share e
    n_significant_mp. Restituisce un DataFrame indicizzato come `returns`
    (le prime `window-1` righe sono NaN, essendo il DGP causale).
    """
    n_assets = returns.shape[1]
    idx = returns.index
    lambda1_share = pd.Series(index=idx, dtype=float)
    n_significant = pd.Series(index=idx, dtype=float)
    _, lambda_plus = marchenko_pastur_bounds(n_assets, window)

    values = returns.values
    for t in range(window - 1, len(idx)):
        window_data = values[t - window + 1: t + 1]
        corr = np.corrcoef(window_data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        eigvals = np.linalg.eigvalsh(corr)  # ascendente
        eigvals = np.clip(eigvals, 0, None)
        total = eigvals.sum()
        lambda1_share.iloc[t] = eigvals[-1] / total if total > 0 else np.nan
        n_significant.iloc[t] = int(np.sum(eigvals > lambda_plus))

    return pd.DataFrame({"lambda1_share": lambda1_share, "n_significant_mp": n_significant})


def _fit_gaussian_hmm_2state(series: pd.Series, random_state: int = 42):
    """Stima (EM, full-sample) un HMM Gaussiano a 2 stati su una serie 1D."""
    from hmmlearn.hmm import GaussianHMM

    x = series.dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=random_state)
    model.fit(x)
    return model


def _forward_filter_probabilities(model, x: np.ndarray) -> np.ndarray:
    """
    Implementa il forward pass (filtraggio) di un HMM Gaussiano già stimato:
    P(stato_t = i | osservazioni_1..t), usando SOLO il passato (nessuno
    smoothing all'indietro, a differenza di `model.predict_proba` di
    hmmlearn che userebbe anche le osservazioni future).
    """
    from scipy.stats import norm

    T = len(x)
    n_states = model.n_components
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())
    A = model.transmat_
    pi = model.startprob_

    alpha = np.zeros((T, n_states))
    emission_0 = np.array([norm.pdf(x[0, 0], means[i], stds[i]) for i in range(n_states)])
    alpha[0] = pi * emission_0
    alpha[0] /= max(alpha[0].sum(), 1e-300)

    for t in range(1, T):
        emission_t = np.array([norm.pdf(x[t, 0], means[i], stds[i]) for i in range(n_states)])
        alpha[t] = (alpha[t - 1] @ A) * emission_t
        s = alpha[t].sum()
        alpha[t] /= s if s > 1e-300 else 1.0

    return alpha  # (T, n_states), righe che sommano a 1


@dataclass
class SpectralRegimeResult:
    features: pd.DataFrame          # lambda1_share, lambda1_share_roc, n_significant_mp, regime_prob_stress
    hmm_stress_state: int
    hmm_means: np.ndarray


def compute_spectral_regime_features(
    returns: pd.DataFrame, window: int = 60, roc_lag: int = 5, fit_hmm: bool = True, random_state: int = 42
) -> SpectralRegimeResult:
    """
    Pipeline completa: spettro rolling -> rate-of-change -> (opzionale) HMM
    a 2 stati con probabilità di regime "stress" filtrate causalmente.
    """
    spectrum = rolling_eigen_spectrum(returns, window)
    lambda1_share = spectrum["lambda1_share"]
    roc = lambda1_share - lambda1_share.shift(roc_lag)

    features = pd.DataFrame({
        "lambda1_share": lambda1_share,
        "lambda1_share_roc": roc,
        "n_significant_mp": spectrum["n_significant_mp"],
    })

    hmm_stress_state = -1
    hmm_means = np.array([])
    if fit_hmm:
        valid = lambda1_share.dropna()
        if len(valid) >= 30:
            model = _fit_gaussian_hmm_2state(valid, random_state=random_state)
            hmm_means = model.means_.flatten()
            hmm_stress_state = int(np.argmax(hmm_means))  # stato con media lambda1_share più alta = "stress"

            alpha = _forward_filter_probabilities(model, valid.values.reshape(-1, 1))
            regime_prob = pd.Series(alpha[:, hmm_stress_state], index=valid.index)
            features["regime_prob_stress"] = regime_prob.reindex(returns.index)
            logger.info(
                "HMM regime spettrale stimato: stato di stress = %d (media lambda1_share=%.3f vs %.3f)",
                hmm_stress_state, hmm_means[hmm_stress_state], hmm_means[1 - hmm_stress_state],
            )
        else:
            features["regime_prob_stress"] = np.nan
            logger.warning("Serie troppo corta per stimare l'HMM di regime spettrale (servono >= 30 osservazioni valide)")
    else:
        features["regime_prob_stress"] = np.nan

    return SpectralRegimeResult(features=features, hmm_stress_state=hmm_stress_state, hmm_means=hmm_means)
