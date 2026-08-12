#!/usr/bin/env python3
"""
scripts/run_pipeline.py
=========================
Orchestratore end-to-end della pipeline:

  dati (Yahoo Finance + Fed + BCE) -> allineamento -> EDA (ACF/autocorr.) ->
  costruzione grafi (MST/PMFG/Granger/Diebold-Yilmaz/settoriale) ->
  modello finale GNARX (media) + DST-ARCH (varianza) ->
  modelli di confronto (GNARX+GNGARCH, AR+GARCH11 non di rete) ->
  backtest rolling out-of-sample -> metriche di accuratezza -> grafici.

Uso
---
    python scripts/run_pipeline.py --synthetic            # demo offline
    python scripts/run_pipeline.py                        # dati reali (richiede rete)
    python scripts/run_pipeline.py --config path/to.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import download as dl
from src.data import synthetic as synth
from src.data import preprocessing as prep
from src.graphs import network_builders as gb
from src.graphs import spectral_regime as specreg
from src.models.gnar import GNAR
from src.models.variance_models import DSTARCH, GNGARCH
from src.models.pipeline import GNARXDSTARCHPipeline, GNARGNGARCHPipeline
from src.models.benchmarks import ARBenchmark, GARCH11Benchmark
from src.evaluation.backtest import rolling_backtest
from src.evaluation.metrics import mean_forecast_metrics, variance_forecast_metrics, var_coverage_test
from src.utils import plotting as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_pipeline")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data(config: dict, use_synthetic: bool):
    if use_synthetic:
        logger.info("Modalità SINTETICA attiva: i dati sono generati localmente da un DGP GNAR+GNGARCH noto "
                     "(nessuna chiamata di rete). Il codice di download reale è comunque disponibile e "
                     "funzionante in `src/data/download.py` per ambienti con accesso a Internet.")
        data, true_graph = synth.generate_synthetic_dataset(config)
        return data, true_graph
    else:
        try:
            data = dl.download_all(config)
            return data, None
        except RuntimeError as exc:
            logger.error("Download dati reali fallito (%s). Rilancia con --synthetic per un run dimostrativo offline.", exc)
            raise


def main():
    parser = argparse.ArgumentParser(description="Pipeline GNARX (media) + DST-ARCH (varianza) su rete finanziaria")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--synthetic", action="store_true", help="Usa dati sintetici invece di Yahoo Finance/FRED/ECB")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    universe = config["data"]["universe"]
    sector_map = dl.sector_map_from_universe(universe)

    # ---------------------------------------------------------------- DATI
    data, true_graph = get_data(config, args.synthetic)
    aligned = prep.align_all_series(data["prices"], data["fed"], data["ecb"])
    returns, fed, ecb = aligned["returns"], aligned["fed"], aligned["ecb"]
    exog = pd.concat([fed, ecb], axis=1)
    exog.columns = ["fed_funds_rate", "ecb_main_refi_rate"]

    # ------------------------------------------------------ REGIME SPETTRALE
    spec_cfg = config.get("spectral_regime", {})
    if spec_cfg.get("enabled", False):
        logger.info("Calcolo dell'indicatore di regime spettrale (autovalori matrice di correlazione rolling)...")
        spec_result = specreg.compute_spectral_regime_features(
            returns, window=spec_cfg["window"], roc_lag=spec_cfg["roc_lag"], fit_hmm=spec_cfg.get("fit_hmm", True),
        )
        spec_features = spec_result.features
        exog = pd.concat([exog, spec_features], axis=1)
        spec_features.to_csv(results_dir / "spectral_regime_features.csv")
        viz.plot_spectral_regime(spec_features, str(results_dir / "spectral_regime.png"))
        logger.info("Feature di regime spettrale aggiunte alle esogene: %s", list(spec_features.columns))

    logger.info("Dataset pronto: %d osservazioni, %d asset (%s -> %s)",
                len(returns), returns.shape[1], returns.index.min().date(), returns.index.max().date())

    # ---------------------------------------------------------------- EDA
    diag = prep.diagnose_series(returns)
    diag.to_csv(results_dir / "eda_diagnostics.csv")
    logger.info("Diagnostica ADF/Ljung-Box salvata in results/eda_diagnostics.csv")

    acf_pacf = prep.compute_acf_pacf(returns, n_lags=20)
    viz.plot_acf_pacf_grid(acf_pacf, str(results_dir / "acf_pacf.png"))

    corr = prep.cross_correlation_matrix(returns)
    corr.to_csv(results_dir / "correlation_matrix.csv")

    # ---------------------------------------------------------------- GRAFI
    logger.info("Costruzione dei grafi di rete...")
    mst = gb.correlation_mst(returns, method=config["graph"]["correlation_method"])
    pmfg = gb.correlation_pmfg(returns, method=config["graph"]["correlation_method"]) if config["graph"]["pmfg"] else None
    granger_g = gb.granger_causality_network(returns, **config["graph"]["granger"])
    dy_graph, dy_matrix, dy_index = gb.diebold_yilmaz_network(returns, **config["graph"]["diebold_yilmaz"])
    sector_g = gb.sector_network(list(returns.columns), sector_map)
    dy_matrix.to_csv(results_dir / "diebold_yilmaz_spillover_matrix.csv")
    logger.info("Diebold-Yilmaz Total Spillover Index: %.2f%%", dy_index)

    viz.plot_graph(mst, "Rete MST (distanza di correlazione)", str(results_dir / "graph_mst.png"), sector_map)
    if pmfg is not None:
        viz.plot_graph(pmfg, "Rete PMFG", str(results_dir / "graph_pmfg.png"), sector_map)
    viz.plot_graph(granger_g, "Rete di causalità di Granger", str(results_dir / "graph_granger.png"), sector_map)
    viz.plot_graph(dy_graph, "Rete di Volatility Spillover (Diebold-Yilmaz)", str(results_dir / "graph_diebold_yilmaz.png"), sector_map)
    viz.plot_graph(sector_g, "Rete settoriale (GICS-like)", str(results_dir / "graph_sector.png"), sector_map)
    if true_graph is not None:
        viz.plot_graph(true_graph, "Grafo VERO usato nel DGP sintetico", str(results_dir / "graph_true_synthetic.png"), sector_map)

    # Grafo per la componente di MEDIA (GNARX): MST, robusto e sparso
    node_order = list(returns.columns)
    max_r_mean = max(config["mean_model"]["gnar_neighbor_orders_R"])
    stage_matrices_mean = gb.build_stage_weight_matrices(mst, node_order, max_r_mean, config["graph"]["neighbor_weighting"])

    # Grafo per la componente di VARIANZA: rete di spillover di volatilità
    # (l'evidenza empirica citata nel documento indica che le reti di
    # spillover sovraperformano quelle basate su correlazione lineare per
    # la previsione della varianza)
    W_variance = dy_matrix.reindex(index=node_order, columns=node_order).values
    stage_matrices_var = gb.build_stage_weight_matrices(dy_graph, node_order, config["variance_model"]["gngarch"]["R"][0], "equal")

    # ---------------------------------------------------------------- MODELLO FINALE (in-sample)
    logger.info("Stima del modello finale GNARX (media) + DST-ARCH (varianza) sull'intero campione...")
    final_pipeline = GNARXDSTARCHPipeline(
        gnar_P=config["mean_model"]["gnar_order_P"],
        gnar_R=config["mean_model"]["gnar_neighbor_orders_R"],
        stage_matrices_mean=stage_matrices_mean,
        exog_lags=config["mean_model"]["exog_lags"],
        dst_ar_order=config["variance_model"]["dst_arch"]["ar_order"],
        dst_log_shift_c=config["variance_model"]["dst_arch"]["log_shift_c"],
        W_variance=W_variance,
    )
    mean_res, var_res = final_pipeline.fit(returns, exog)
    logger.info("GNARX in-sample R2 (pooled su tutti i nodi): %.4f | osservazioni usate: %d", mean_res.r_squared, mean_res.n_obs_used)
    logger.info("DST-ARCH: rho(spaziale)=%.4f, phi(temporale)=%.4f, convergenza=%s", var_res.rho, var_res.phi, var_res.converged)

    coef_summary = pd.Series(mean_res.coefficients, index=mean_res.coef_names)
    coef_summary["__intercept__"] = mean_res.intercept
    coef_summary.to_csv(results_dir / "gnarx_coefficients.csv", header=["value"])

    # ---------------------------------------------------------------- BACKTEST ROLLING
    bt_cfg = config["backtest"]
    train_window = bt_cfg["train_window"]
    test_horizon = bt_cfg["test_horizon"]
    refit_every = bt_cfg["refit_every"]

    def build_final(Y_train, exog_train):
        m = GNARXDSTARCHPipeline(
            gnar_P=config["mean_model"]["gnar_order_P"],
            gnar_R=config["mean_model"]["gnar_neighbor_orders_R"],
            stage_matrices_mean=stage_matrices_mean,
            exog_lags=config["mean_model"]["exog_lags"],
            dst_ar_order=config["variance_model"]["dst_arch"]["ar_order"],
            dst_log_shift_c=config["variance_model"]["dst_arch"]["log_shift_c"],
            W_variance=W_variance,
        )
        m.fit(Y_train, exog_train)
        return m

    def predict_final(model, Y_history, exog_history):
        f = model.predict_one_step(Y_history, exog_history)
        return f.mean, f.variance

    logger.info("Backtest rolling: modello finale GNARX + DST-ARCH (%d passi, refit ogni %d)...", test_horizon, refit_every)
    bt_final = rolling_backtest(
        returns, exog, build_final, predict_final,
        train_window=train_window, test_horizon=test_horizon, refit_every=refit_every,
        max_lag_needed=max(config["mean_model"]["gnar_order_P"], max(config["mean_model"]["exog_lags"])),
    )

    def build_gngarch(Y_train, exog_train):
        m = GNARGNGARCHPipeline(
            gnar_P=config["mean_model"]["gnar_order_P"],
            gnar_R=config["mean_model"]["gnar_neighbor_orders_R"],
            stage_matrices_mean=stage_matrices_mean,
            exog_lags=config["mean_model"]["exog_lags"],
            gngarch_R=config["variance_model"]["gngarch"]["R"][0],
            stage_matrices_var=stage_matrices_var,
            distribution=config["variance_model"]["gngarch"]["distribution"],
        )
        m.fit(Y_train, exog_train)
        return m

    logger.info("Backtest rolling: modello di confronto GNARX + GNGARCH...")
    bt_gngarch = rolling_backtest(
        returns, exog, build_gngarch, predict_final,
        train_window=train_window, test_horizon=test_horizon, refit_every=refit_every,
        max_lag_needed=max(config["mean_model"]["gnar_order_P"], max(config["mean_model"]["exog_lags"])),
    )

    def build_benchmark(Y_train, exog_train):
        ar = ARBenchmark(lags=config["mean_model"]["gnar_order_P"]).fit(Y_train)
        resid = pd.DataFrame({c: Y_train[c] - ar._fitted[c].fittedvalues.reindex(Y_train.index) for c in Y_train.columns}).dropna()
        garch = GARCH11Benchmark(distribution="t").fit(resid)
        return {"ar": ar, "garch": garch, "last_h": garch.conditional_variance().iloc[-1]}

    def predict_benchmark(model, Y_history, exog_history):
        mean_pred = model["ar"].predict_one_step(Y_history)
        var_pred = model["garch"].forecast_one_step()
        return mean_pred, var_pred

    logger.info("Backtest rolling: benchmark non di rete AR + GARCH(1,1)...")
    bt_bench = rolling_backtest(
        returns, exog, build_benchmark, predict_benchmark,
        train_window=train_window, test_horizon=test_horizon, refit_every=refit_every,
        max_lag_needed=config["mean_model"]["gnar_order_P"],
    )

    # ---------------------------------------------------------------- METRICHE
    logger.info("Calcolo delle metriche di accuratezza...")
    realized_sq = bt_final.actual_returns ** 2

    results_summary = {}
    for name, bt in [("GNARX_DSTARCH_finale", bt_final), ("GNARX_GNGARCH", bt_gngarch), ("AR_GARCH11_benchmark", bt_bench)]:
        mean_m = mean_forecast_metrics(bt.actual_returns, bt.mean_forecasts)
        var_m = variance_forecast_metrics(bt.actual_returns ** 2, bt.variance_forecasts)
        var_cov = var_coverage_test(bt.actual_returns, bt.mean_forecasts, bt.variance_forecasts)
        mean_m.to_csv(results_dir / f"metrics_mean_{name}.csv")
        var_m.to_csv(results_dir / f"metrics_variance_{name}.csv")
        var_cov.to_csv(results_dir / f"metrics_var_coverage_{name}.csv")
        results_summary[name] = {
            "RMSE_media": mean_m.loc["__ALL__", "RMSE"],
            "Directional_Accuracy": mean_m.loc["__ALL__", "Directional_Accuracy"],
            "QLIKE_varianza": var_m.loc["__ALL__", "QLIKE"],
            "MSE_varianza": var_m.loc["__ALL__", "MSE_variance"],
        }

    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv(results_dir / "metrics_summary_comparison.csv")
    logger.info("\n%s", summary_df.to_string())

    # ---------------------------------------------------------------- GRAFICI FINALI
    example_ticker = node_order[0]
    viz.plot_rolling_mean_forecast(bt_final.actual_returns, bt_final.mean_forecasts, example_ticker, str(results_dir / "rolling_mean_forecast_example.png"))
    viz.plot_rolling_volatility_forecast(bt_final.actual_returns ** 2, bt_final.variance_forecasts, example_ticker, str(results_dir / "rolling_volatility_forecast_example.png"))
    mean_m_final = mean_forecast_metrics(bt_final.actual_returns, bt_final.mean_forecasts)
    viz.plot_metrics_bar(mean_m_final, "RMSE", "RMSE previsione media per asset - Modello finale", str(results_dir / "rmse_by_asset.png"))

    logger.info("Pipeline completata. Tutti gli output sono in: %s", results_dir)


if __name__ == "__main__":
    main()
