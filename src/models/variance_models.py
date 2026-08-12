"""
src/models/variance_models.py
===============================
Implementa i modelli di varianza condizionata su rete descritti nel
documento di riferimento:

  - GNGARCH(P,Q,[R_p],[M_q]): estensione network del GARCH, con termini
    ARCH/GARCH "di rete" oltre a quelli diretti. Versione "globale"
    (alpha, alpha_net, beta, beta_net condivisi tra i nodi; omega_i
    specifico per nodo, per permettere livelli di varianza incondizionata
    eterogenei). Stimato per Quasi-Maximum Likelihood numerica.
  - TGNGARCH: estensione con effetto leva (soglia) sia diretto che di rete.
  - DST-ARCH (Dynamic Spatiotemporal ARCH): rappresentazione ARMA per la
    log-volatilità su rete, resa trattabile diagonalizzando l'operatore
    spaziale tramite la trasformazione ortonormale data dagli autovettori
    della matrice di adiacenza simmetrizzata (W_sym = U Lambda U'), e stimato
    con un criterio dei minimi quadrati sul sistema disaccoppiato nel dominio
    spettrale (approccio a minima distanza, nello spirito del GMM descritto
    nel documento).

    NOTA METODOLOGICA: il documento descrive il DST-ARCH ad alto livello
    (rappresentazione ARMA in log-volatilità, interazione spaziale
    contemporanea, stima GMM via trasformazioni ortonormali) senza fornire
    lo stimatore esatto della pubblicazione originale. L'implementazione qui
    presente segue fedelmente questi principi metodologici con uno stimatore
    a minima distanza pienamente specificato e riproducibile, ma non
    pretende di replicare bit-a-bit l'estimatore di uno specifico paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t


# ==========================================================================
# GNGARCH / TGNGARCH
# ==========================================================================

@dataclass
class GNGARCHResult:
    omega: np.ndarray            # (N,) intercette specifiche per nodo
    alpha: float
    alpha_net: List[float]       # per ordine di vicinato r=1..R
    beta: float
    beta_net: List[float]        # per ordine di vicinato q=1..M (qui M=1 semplificato)
    gamma: Optional[float]       # solo TGNGARCH (effetto leva diretto)
    gamma_net: Optional[List[float]]  # solo TGNGARCH (effetto leva di rete)
    log_likelihood: float
    conditional_variance: pd.DataFrame
    converged: bool


class GNGARCH:
    """
    GNGARCH(P=1,Q=1,[R],[M=1]) globale con innovazioni Normali o t-Student.
    Per semplicità e identificabilità (coerente con la pratica dei modelli
    Network-GARCH globali) si fissano P=Q=1 con effetto di rete al solo
    ritardo 1; R (ordine massimo di vicinato) è configurabile.

    threshold=True attiva la variante TGNGARCH (effetto leva diretto + rete).
    """

    def __init__(self, R: int = 1, distribution: str = "t", threshold: bool = False):
        self.R = R
        self.distribution = distribution
        self.threshold = threshold
        self.result_: Optional[GNGARCHResult] = None
        self._S: List[np.ndarray] = []
        self._node_order: List[str] = []

    def fit(self, eps: pd.DataFrame, stage_matrices: List[np.ndarray]) -> GNGARCHResult:
        self._S = stage_matrices[: self.R]
        self._node_order = list(eps.columns)
        E = eps.values  # (T, N)
        T, N = E.shape
        neg_mask = (E < 0).astype(float)

        n_net = self.R
        # vettore parametri: [omega(N), alpha, alpha_net(n_net), beta, beta_net(n_net),
        #                      (gamma, gamma_net(n_net) se threshold), nu (se t-Student)]
        uncond_var = np.var(E, axis=0)
        x0 = list(0.05 * uncond_var) + [0.05] + [0.03] * n_net + [0.85] + [0.02] * n_net
        bounds = [(1e-10, None)] * N + [(0.0, 1.0)] + [(0.0, 1.0)] * n_net + [(0.0, 0.999)] + [(0.0, 1.0)] * n_net
        if self.threshold:
            x0 += [0.02] + [0.01] * n_net
            bounds += [(0.0, 1.0)] + [(0.0, 1.0)] * n_net
        if self.distribution == "t":
            x0 += [8.0]
            bounds += [(2.1, 60.0)]

        def unpack(x):
            i = 0
            omega = np.array(x[i:i + N]); i += N
            alpha = x[i]; i += 1
            alpha_net = x[i:i + n_net]; i += n_net
            beta = x[i]; i += 1
            beta_net = x[i:i + n_net]; i += n_net
            gamma, gamma_net = None, None
            if self.threshold:
                gamma = x[i]; i += 1
                gamma_net = x[i:i + n_net]; i += n_net
            nu = x[i] if self.distribution == "t" else None
            return omega, alpha, alpha_net, beta, beta_net, gamma, gamma_net, nu

        def neg_log_lik(x):
            omega, alpha, alpha_net, beta, beta_net, gamma, gamma_net, nu = unpack(x)
            h = np.zeros((T, N))
            h[0] = np.maximum(uncond_var, 1e-10)
            eps2 = E ** 2
            for t in range(1, T):
                base = omega + alpha * eps2[t - 1] + beta * h[t - 1]
                for r in range(1, n_net + 1):
                    Sr = self._S[r - 1]
                    base = base + alpha_net[r - 1] * (Sr @ eps2[t - 1]) + beta_net[r - 1] * (Sr @ h[t - 1])
                if self.threshold:
                    base = base + gamma * eps2[t - 1] * neg_mask[t - 1]
                    for r in range(1, n_net + 1):
                        Sr = self._S[r - 1]
                        base = base + gamma_net[r - 1] * (Sr @ (eps2[t - 1] * neg_mask[t - 1]))
                h[t] = np.maximum(base, 1e-10)

            h_valid = h[1:]
            e_valid = E[1:]
            if self.distribution == "normal":
                ll = -0.5 * np.sum(np.log(2 * np.pi * h_valid) + e_valid ** 2 / h_valid)
            else:
                std_e = e_valid / np.sqrt(h_valid)
                log_pdf = student_t.logpdf(std_e, df=nu) - 0.5 * np.log(h_valid)
                ll = np.sum(log_pdf)
            if not np.isfinite(ll):
                return 1e12
            return -ll

        opt = minimize(neg_log_lik, x0=np.array(x0), method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 200, "ftol": 1e-8})

        omega, alpha, alpha_net, beta, beta_net, gamma, gamma_net, nu = unpack(opt.x)

        # ricalcola h con parametri ottimali per restituirla
        h = np.zeros((T, N))
        h[0] = np.maximum(uncond_var, 1e-10)
        eps2 = E ** 2
        for t in range(1, T):
            base = omega + alpha * eps2[t - 1] + beta * h[t - 1]
            for r in range(1, n_net + 1):
                Sr = self._S[r - 1]
                base = base + alpha_net[r - 1] * (Sr @ eps2[t - 1]) + beta_net[r - 1] * (Sr @ h[t - 1])
            if self.threshold:
                base = base + gamma * eps2[t - 1] * neg_mask[t - 1]
                for r in range(1, n_net + 1):
                    Sr = self._S[r - 1]
                    base = base + gamma_net[r - 1] * (Sr @ (eps2[t - 1] * neg_mask[t - 1]))
            h[t] = np.maximum(base, 1e-10)

        self.result_ = GNGARCHResult(
            omega=omega, alpha=float(alpha), alpha_net=list(alpha_net),
            beta=float(beta), beta_net=list(beta_net),
            gamma=float(gamma) if gamma is not None else None,
            gamma_net=list(gamma_net) if gamma_net is not None else None,
            log_likelihood=-float(opt.fun),
            conditional_variance=pd.DataFrame(h, index=eps.index, columns=eps.columns),
            converged=bool(opt.success),
        )
        self._last_params = opt.x
        return self.result_

    def forecast_one_step(self, eps_last: np.ndarray, h_last: np.ndarray) -> np.ndarray:
        """Prevede h_{t+1} dati eps_t e h_t (array (N,))."""
        assert self.result_ is not None
        r = self.result_
        base = r.omega + r.alpha * eps_last ** 2 + r.beta * h_last
        for k in range(1, self.R + 1):
            Sr = self._S[k - 1]
            base = base + r.alpha_net[k - 1] * (Sr @ eps_last ** 2) + r.beta_net[k - 1] * (Sr @ h_last)
        if self.threshold:
            neg = (eps_last < 0).astype(float)
            base = base + r.gamma * eps_last ** 2 * neg
            for k in range(1, self.R + 1):
                Sr = self._S[k - 1]
                base = base + r.gamma_net[k - 1] * (Sr @ (eps_last ** 2 * neg))
        return np.maximum(base, 1e-10)


# ==========================================================================
# DST-ARCH: log-volatilità ARMA su rete, diagonalizzata via trasformazione
# ortonormale (autovettori della matrice di adiacenza simmetrizzata)
# ==========================================================================

@dataclass
class DSTARCHResult:
    rho: float                  # coefficiente spaziale simultaneo
    phi: float                  # coefficiente autoregressivo temporale
    bias_correction: float      # correzione moltiplicativa per il ritorno da log a livello
    log_variance: pd.DataFrame  # xi_t stimata (log-varianza fitted)
    conditional_variance: pd.DataFrame
    eigenvalues: np.ndarray
    converged: bool


class DSTARCH:
    """
    Modello Dynamic Spatiotemporal ARCH semplificato:

        xi_t = rho * W_sym @ xi_t + phi * xi_{t-1} + u_t,   xi_t = log(eps_t^2 + c)

    Riscritto come (I - rho*W_sym) xi_t = phi*xi_{t-1} + u_t. Diagonalizzando
    W_sym = U Lambda U' (U ortonormale, autovalori reali essendo W_sym
    simmetrica), nel dominio trasformato z_t = U' xi_t il sistema si
    disaccoppia in N AR(1) indipendenti:

        z_{k,t} = [phi / (1 - rho*lambda_k)] * z_{k,t-1} + v_{k,t}

    (rho, phi) sono stimati minimizzando la somma dei quadrati dei residui
    v_{k,t} su tutti i k e t simultaneamente (stimatore a minima distanza,
    equivalente alle condizioni di momento GMM E[v_{k,t} z_{k,t-1}] = 0).
    """

    def __init__(self, ar_order: int = 1, log_shift_c: float = 1e-6, symmetrize: bool = True):
        assert ar_order == 1, "Questa implementazione supporta ar_order=1 (estendibile ad ARMA(p,q))"
        self.log_shift_c = log_shift_c
        self.symmetrize = symmetrize
        self.result_: Optional[DSTARCHResult] = None
        self._U: Optional[np.ndarray] = None
        self._eigvals: Optional[np.ndarray] = None
        self._node_order: List[str] = []

    def fit(self, eps: pd.DataFrame, W: np.ndarray) -> DSTARCHResult:
        self._node_order = list(eps.columns)
        W_use = (W + W.T) / 2.0 if self.symmetrize else W
        row_sums = W_use.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_sym = W_use / row_sums
        W_sym = (W_sym + W_sym.T) / 2.0  # forza simmetria numerica esatta per eigh

        eigvals, U = np.linalg.eigh(W_sym)
        self._U = U
        self._eigvals = eigvals

        xi = np.log(eps.values ** 2 + self.log_shift_c)  # (T, N)
        Z = xi @ U  # z_t = U' xi_t  (equivalente, righe=tempo)
        Z_lag = np.vstack([np.full((1, Z.shape[1]), np.nan), Z[:-1]])

        max_abs_lambda = np.max(np.abs(eigvals))
        rho_bound = 0.98 / max(max_abs_lambda, 1e-6)

        def objective(params):
            rho, phi = params
            denom = 1 - rho * eigvals
            if np.any(np.abs(denom) < 1e-4):
                return 1e12
            phi_k = phi / denom  # (N,)
            pred = Z_lag[1:] * phi_k[np.newaxis, :]
            resid = Z[1:] - pred
            return float(np.sum(resid ** 2))

        best = None
        for rho0 in np.linspace(-0.8 * rho_bound, 0.8 * rho_bound, 5):
            opt = minimize(
                objective, x0=np.array([rho0, 0.3]), method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500},
            )
            if best is None or opt.fun < best.fun:
                best = opt

        rho_hat, phi_hat = best.x
        rho_hat = float(np.clip(rho_hat, -0.98 * rho_bound, 0.98 * rho_bound))

        denom = 1 - rho_hat * eigvals
        phi_k = phi_hat / denom
        Z_fitted = np.vstack([Z[:1], Z_lag[1:] * phi_k[np.newaxis, :]])
        xi_fitted = Z_fitted @ U.T  # ritorna al dominio originale (U ortonormale => U^-1 = U')

        eps2_actual_mean = float(np.mean(eps.values ** 2))
        h_raw = np.exp(xi_fitted)
        bias_correction = eps2_actual_mean / max(float(np.mean(h_raw)), 1e-12)
        h = h_raw * bias_correction

        self.result_ = DSTARCHResult(
            rho=rho_hat, phi=float(phi_hat), bias_correction=bias_correction,
            log_variance=pd.DataFrame(xi_fitted, index=eps.index, columns=eps.columns),
            conditional_variance=pd.DataFrame(np.maximum(h, 1e-10), index=eps.index, columns=eps.columns),
            eigenvalues=eigvals, converged=bool(best.success),
        )
        return self.result_

    def forecast_one_step(self, eps_last: np.ndarray) -> np.ndarray:
        """Prevede h_{t+1} dato eps_t (array (N,))."""
        assert self.result_ is not None and self._U is not None
        xi_last = np.log(eps_last ** 2 + self.log_shift_c)
        z_last = self._U.T @ xi_last
        denom = 1 - self.result_.rho * self._eigvals
        phi_k = self.result_.phi / denom
        z_next = phi_k * z_last
        xi_next = self._U @ z_next
        h_next = np.exp(xi_next) * self.result_.bias_correction
        return np.maximum(h_next, 1e-10)
