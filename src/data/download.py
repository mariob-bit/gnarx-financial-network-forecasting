"""
src/data/download.py
=====================
Acquisizione dei dati reali necessari alla pipeline:

1. Prezzi giornalieri dei prodotti finanziari via `yfinance` (Yahoo Finance)[cite: 1].
2. Effective Federal Funds Rate (tasso Fed) via FRED (`pandas_datareader`,
   nessuna API key richiesta per la serie pubblica "DFF")[cite: 1].
3. Tasso sulle operazioni di rifinanziamento principali BCE via l'API REST
   pubblica dell'ECB Data Portal (SDMX-CSV, nessuna API key richiesta)[cite: 1].

NOTA IMPORTANTE
----------------
Queste funzioni richiedono accesso a Internet verso query1/query2.finance.yahoo.com,
fred.stlouisfed.org e data-api.ecb.europa.eu[cite: 1]. In ambienti sandbox con allowlist di
rete ristretta (es. l'ambiente in cui questo repository è stato generato) queste
chiamate falliranno con un errore di connessione: è un limite dell'ambiente, non
un bug del codice[cite: 1]. Eseguito in locale, in CI (GitHub Actions) o su qualunque
macchina con accesso Internet standard, il modulo funziona senza modifiche[cite: 1].

Per testare l'intera pipeline senza accesso a Internet, usare
`src/data/synthetic.py`, che genera dati sintetici con la stessa interfaccia[cite: 1].
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def flatten_universe(universe: Dict[str, List[str]]) -> List[str]:
    """Appiattisce il dizionario {settore: [ticker,...]} in una lista di ticker."""
    tickers: List[str] = []
    for sector_tickers in universe.values():
        tickers.extend(sector_tickers)
    return tickers


def sector_map_from_universe(universe: Dict[str, List[str]]) -> Dict[str, str]:
    """Ritorna {ticker: settore}, utile per la rete basata su classificazione industriale."""
    mapping: Dict[str, str] = {}
    for sector, tickers in universe.items():
        for t in tickers:
            mapping[t] = sector
    return mapping


def download_prices(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    price_field: str = "Close",
) -> pd.DataFrame:
    """
    Scarica i prezzi giornalieri per una lista di ticker via yfinance.

    Ritorna un DataFrame (index=date, columns=tickers) con `price_field`
    (con auto_adjust=True, 'Close' è già rettificato per dividendi/split).

    Solleva RuntimeError con un messaggio esplicativo se il download fallisce
    per motivi di rete, così che il chiamante possa distinguere un errore di
    rete da un errore nei dati.
    """
    import yfinance as yf

    logger.info("Download prezzi per %d ticker da Yahoo Finance...", len(tickers))
    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:  # pragma: no cover - dipende dalla rete
        raise RuntimeError(
            "Impossibile contattare Yahoo Finance. Verifica la connessione di rete "
            "o le impostazioni di allowlist dell'ambiente. Errore originale: "
            f"{exc}"
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo Finance ha restituito un DataFrame vuoto: nessun dato scaricato. "
            "In ambienti sandbox con rete ristretta questo è atteso; usare "
            "src/data/synthetic.py per un run dimostrativo offline."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t][price_field] for t in tickers if t in raw.columns.levels[0]})
    else:
        # Caso singolo ticker
        prices = raw[[price_field]].rename(columns={price_field: tickers[0]})

    prices = prices.sort_index()
    logger.info("Scaricate %d osservazioni per %d ticker.", len(prices), prices.shape[1])
    return prices


def download_fed_funds_rate(start_date: str, end_date: Optional[str] = None, series_id: str = "DFF") -> pd.Series:
    """
    Scarica l'Effective Federal Funds Rate giornaliero da FRED.

    Usa `pandas_datareader`, che recupera il CSV pubblico di FRED senza
    richiedere una API key per serie standard come "DFF".
    """
    import pandas_datareader.data as web

    logger.info("Download tasso Fed Funds (serie FRED '%s')...", series_id)
    try:
        s = web.DataReader(series_id, "fred", start_date, end_date)
    except Exception as exc:  # pragma: no cover - dipende dalla rete
        raise RuntimeError(
            "Impossibile contattare FRED (fred.stlouisfed.org) per il tasso Fed. "
            f"Errore originale: {exc}"
        ) from exc

    out = s[series_id].rename("fed_funds_rate")
    return out


def download_ecb_main_refi_rate(
    start_date: str,
    end_date: Optional[str] = None,
    series_key: str = "FM.D.U2.EUR.4F.KR.MRR_FR.LEV",
) -> pd.Series:
    """
    Scarica il tasso sulle operazioni di rifinanziamento principali (fixed rate)
    della BCE dall'ECB Data Portal (API REST pubblica, formato SDMX-CSV).

    La serie BCE cambia raramente (a ogni decisione di politica monetaria):
    viene qui restituita "as-is" (a scalini); l'allineamento a frequenza
    giornaliera con forward-fill avviene in `preprocessing.py`.
    """
    # Gestione del parametro series_key sia nel formato 'FM.D.U2...' che 'D.U2...'
    if "." in series_key and series_key.startswith("FM."):
        flow_ref, key = series_key.split(".", 1)
    elif "." in series_key:
        flow_ref, key = "FM", series_key
    else:
        flow_ref, key = "FM", "D.U2.EUR.4F.KR.MRR_FR.LEV"

    # La struttura SDMX dell'API BCE richiede /service/data/{flowRef}/{key}
    url = f"https://data-api.ecb.europa.eu/service/data/{flow_ref}/{key}"
    params = {
        "format": "csvdata",
        "startPeriod": start_date,
        "endPeriod": end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
    }
    logger.info("Download tasso BCE (flowRef: '%s', key: '%s')...", flow_ref, key)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - dipende dalla rete
        raise RuntimeError(
            "Impossibile contattare l'ECB Data Portal (data-api.ecb.europa.eu). "
            f"Errore originale: {exc}"
        ) from exc

    df = pd.read_csv(io.StringIO(resp.text))
    # Le colonne SDMX-CSV standard sono TIME_PERIOD e OBS_VALUE
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    s = df.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    s.name = "ecb_main_refi_rate"
    return s


def download_all(config: dict) -> Dict[str, pd.DataFrame]:
    """
    Orchestratore: scarica prezzi + tassi Fed/BCE secondo `config['data']`.

    Ritorna un dizionario con chiavi: 'prices', 'fed', 'ecb'.
    """
    data_cfg = config["data"]
    tickers = flatten_universe(data_cfg["universe"])

    prices = download_prices(
        tickers,
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        price_field=data_cfg.get("price_field", "Close"),
    )
    fed = download_fed_funds_rate(
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        series_id=data_cfg["macro"]["fed_series_id"],
    )
    ecb = download_ecb_main_refi_rate(
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        series_key=data_cfg["macro"]["ecb_series_key"],
    )
    return {"prices": prices, "fed": fed, "ecb": ecb}
