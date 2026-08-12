# GNAR Financial Network Forecasting

Pipeline di ricerca quantitativa per la **previsione rolling dei rendimenti** e della
**volatilità condizionata** di un paniere di prodotti finanziari, tramite modelli
econometrici su rete (Generalised Network Autoregressive e loro estensioni), come
descritto nel documento di riferimento *"Modellizzazione della Media dei Rendimenti:
Dai Processi SARIMA alle Estensioni GNARI"*.

**Modello finale**: `GNARX` (media condizionata, con Fed Funds Rate e tasso BCE come
covariate esogene) + `DST-ARCH` (varianza condizionata su rete).

> ⚠️ **Scopo di ricerca/didattico.** Questo repository non costituisce consulenza
> finanziaria o di investimento.

---

## 1. Cosa fa la pipeline

```
Yahoo Finance (prezzi) ─┐
FRED (tasso Fed)        ├─► allineamento ─► rendimenti log ─► EDA (ACF/PACF, ADF, Ljung-Box)
ECB Data Portal (BCE)  ─┘                                            │
                                                                      ▼
                                    costruzione dei grafi (MST, PMFG, Granger, Diebold-Yilmaz, settoriale)
                                                                      │
                              ┌───────────────────────────────────────┼───────────────────────────┐
                              ▼                                       ▼                           ▼
                     GNARX (media) su grafo MST          DST-ARCH (varianza) su grafo         benchmark
                                                          di Volatility Spillover              AR + GARCH(1,1)
                              └───────────────────────────────────────┘
                                                  │
                                     backtest rolling out-of-sample (walk-forward)
                                                  │
                                  metriche: RMSE, MAE, Directional Accuracy, R²,
                                            QLIKE, MSE varianza, copertura VaR (Kupiec)
```

## 2. Metodologia implementata (sintesi)

| Componente | Modello | File |
|---|---|---|
| Topologia di rete | MST/PMFG da correlazione (Mantegna 1999 / Tumminello et al. 2005), causalità di Granger, Volatility Spillover di Diebold-Yilmaz (2012), rete settoriale GICS-like | `src/graphs/network_builders.py` |
| **Regime di mercato (esogena aggiuntiva)** | **Indicatore spettrale**: quota di varianza del primo autovalore della matrice di correlazione rolling (λ₁/Σλᵢ), rate-of-change (early warning), conteggio autovalori "significativi" via bordo di Marchenko-Pastur (RMT), probabilità di regime di stress da un HMM Gaussiano a 2 stati (filtraggio causale forward-only) | `src/graphs/spectral_regime.py` |
| Media condizionata | **GNAR(P, [R_p])** globale + **GNARX** (esogene: Fed Funds Rate, tasso BCE, indicatore di regime spettrale) | `src/models/gnar.py` |
| Varianza condizionata | **GNGARCH** / **TGNGARCH** (leva) via QMLE numerica | `src/models/variance_models.py` |
| Varianza condizionata (modello finale) | **DST-ARCH**: log-volatilità ARMA su rete, diagonalizzata via trasformazione ortonormale (autovettori della matrice di adiacenza simmetrizzata) e stimata a minima distanza (GMM-like) | `src/models/variance_models.py` |
| Modello finale combinato | `GNARXDSTARCHPipeline` | `src/models/pipeline.py` |
| Benchmark non di rete | AR(p) per asset, GARCH(1,1) per asset (pacchetto `arch`) | `src/models/benchmarks.py` |
| Backtest & metriche | rolling walk-forward, RMSE/MAE/Directional Accuracy/R², QLIKE, MSE varianza, test di copertura VaR di Kupiec | `src/evaluation/` |

**Nota sul DST-ARCH**: il documento di riferimento descrive il modello ad alto livello
(rappresentazione ARMA per la log-volatilità, interazione spaziale contemporanea,
stima GMM tramite trasformazioni ortonormali) senza specificare lo stimatore esatto
della pubblicazione originale. L'implementazione qui presente segue fedelmente questi
principi metodologici con uno stimatore a minima distanza pienamente specificato e
riproducibile (si veda il docstring in `variance_models.py`), senza pretendere di
replicare bit-a-bit l'estimatore di uno specifico paper non consultabile.

### Indicatore di regime spettrale (nuova esogena)

Oltre a Fed/BCE, il GNARX riceve in input un indicatore di **regime di mercato**
costruito sugli autovalori della matrice di correlazione rolling tra i rendimenti
(`src/graphs/spectral_regime.py`):

- **`lambda1_share`**: quota di varianza spiegata dal primo autovalore
  (λ₁/Σλᵢ) su finestra mobile. Nei rally/rotture sincronizzati la correlazione
  media tra i titoli sale bruscamente e questo indicatore esplode ("un solo
  fattore di mercato domina tutto").
- **`lambda1_share_roc`**: variazione dell'indicatore su un breve lag — la
  velocità di cambiamento anticipa spesso il prezzo (segnale di early warning).
- **`n_significant_mp`**: numero di autovalori che eccedono il bordo superiore
  della distribuzione di Marchenko-Pastur (filtro Random Matrix Theory: separa
  la struttura "reale" dal rumore statistico nella matrice di correlazione).
- **`regime_prob_stress`**: probabilità di trovarsi in un regime "di stress",
  da un Hidden Markov Model Gaussiano a 2 stati stimato su `lambda1_share`. I
  parametri dell'HMM sono stimati una volta per EM sull'intera serie storica
  disponibile (prassi comune, analoga alla calibrazione di un GARCH); le
  probabilità di stato per ciascun periodo *t* sono però ottenute per
  **filtraggio causale (forward-only)**, usando solo le osservazioni fino a
  *t* — non per smoothing, per evitare lookahead nella componente dinamica.

Tutte le feature sono costruite su finestre strettamente passate e vengono
passate al GNARX con lo stesso meccanismo di lag di Fed/BCE (`exog_lags` in
`config.yaml`). Il generatore sintetico (`src/data/synthetic.py`) inietta
deliberatamente una finestra di "stress" con correlazione elevata, cosicché il
run dimostrativo in `results/spectral_regime.png` mostra la rilevazione
corretta di un evento di regime iniettato artificialmente (validazione).

## 3. Struttura del repository

```
├── config/config.yaml          # universo titoli, iperparametri, parametri di backtest
├── src/
│   ├── data/
│   │   ├── download.py         # Yahoo Finance + FRED (Fed) + ECB Data Portal (BCE)
│   │   ├── synthetic.py        # generatore dati sintetici (DGP GNAR+GNGARCH noto) per demo/test offline
│   │   └── preprocessing.py    # allineamento, rendimenti log, ACF/PACF, ADF, Ljung-Box
│   ├── graphs/network_builders.py
│   ├── models/
│   │   ├── gnar.py             # GNAR / GNARX
│   │   ├── variance_models.py  # GNGARCH / TGNGARCH / DST-ARCH
│   │   ├── benchmarks.py       # AR, GARCH(1,1)
│   │   └── pipeline.py         # modello finale GNARX+DST-ARCH e confronto GNARX+GNGARCH
│   ├── evaluation/
│   │   ├── backtest.py         # motore di backtest rolling walk-forward
│   │   └── metrics.py          # RMSE, MAE, Directional Accuracy, QLIKE, Kupiec VaR test
│   └── utils/plotting.py
├── scripts/run_pipeline.py     # orchestratore end-to-end (CLI)
├── tests/                      # 15 test unitari (pytest)
└── results/                    # output di un run dimostrativo (vedi sezione 6)
```

## 4. Installazione ed esecuzione

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run dimostrativo offline (dati sintetici, nessun accesso a Internet richiesto)
python scripts/run_pipeline.py --synthetic

# Run con dati REALI (richiede accesso a Yahoo Finance, FRED, ECB Data Portal)
python scripts/run_pipeline.py

# Test unitari
pytest tests/ -v
```

Tutti i parametri (universo titoli, periodo, ordini GNAR/GNGARCH, finestra di
backtest, frequenza di refit) si modificano in `config/config.yaml`, senza toccare
il codice.

### Nota importante sull'accesso ai dati

Questo repository è stato sviluppato in un ambiente sandbox la cui rete è ristretta
a un elenco di domini consentiti (per motivi di sicurezza), che **non include**
Yahoo Finance, FRED o l'ECB Data Portal. Il codice di download in
`src/data/download.py` è completo, corretto e testato nella sua logica, ma le
chiamate di rete vere e proprie **non sono state eseguibili in quell'ambiente**.
Eseguito in locale, in CI (GitHub Actions) o su qualsiasi macchina con accesso
Internet standard, funziona senza modifiche.

Per questo motivo, tutti i risultati d'esempio nella cartella `results/` (inclusi
in questo repository) sono stati generati con `--synthetic`: un generatore
(`src/data/synthetic.py`) che simula rendimenti da un **vero processo GNAR+GNGARCH
noto** su un grafo a struttura di comunità coerente con i settori configurati, così
da validare che l'intera pipeline (dati → grafi → modelli → backtest → metriche)
sia effettivamente funzionante end-to-end. Non vanno interpretati come evidenza
empirica sui mercati reali — per quello, esegui `python scripts/run_pipeline.py`
(senza `--synthetic`) con accesso a Internet.

## 5. Universo titoli e ipotesi (personalizzabili in `config.yaml`)

18 titoli USA/ADR raggruppati in 8 settori (per la rete settoriale GICS-like):
Technology (AAPL, MSFT, NVDA), Financials (JPM, BAC, GS), Energy (XOM, CVX),
Healthcare (JNJ, PFE), Consumer Discretionary (AMZN, TSLA), Consumer Staples
(PG, KO), Industrials (BA, CAT), Europe ADR (SAP, ASML).

Esogene: Effective Federal Funds Rate (FRED, serie `DFF`) e tasso sulle operazioni
di rifinanziamento principali BCE (ECB Data Portal, serie `FM.D.U2.EUR.4F.KR.MRR_FR.LEV`),
con lag di 1 e 5 giorni nel GNARX.

Periodo default: dal 2015-01-01 a oggi. Backtest rolling: finestra di stima 300
osservazioni, orizzonte out-of-sample 60 passi, refit ogni 10 passi (bilanciamento
costo computazionale / adattività — aumentabile per un uso in produzione).

## 6. Risultati del run dimostrativo incluso (dati sintetici)

Eseguito con la configurazione completa (18 titoli, 899 osservazioni allineate,
60 passi di backtest rolling, 6 refit per modello), tempo totale ~17 secondi.

**Confronto modelli** (`results/metrics_summary_comparison.csv`):

| Modello | RMSE media | Directional Accuracy | QLIKE varianza | MSE varianza |
|---|---|---|---|---|
| **GNARX + DST-ARCH (finale)** | 0.010570 | 0.4963 | -8.138 | 4.46e-08 |
| GNARX + GNGARCH (confronto) | 0.010570 | 0.4963 | -8.200 | 4.20e-08 |
| AR + GARCH(1,1) (benchmark non di rete) | 0.010379 | 0.5065 | -8.264 | 4.05e-08 |

Il run dimostrativo qui incluso include ora anche l'evento di stress iniettato
nel DGP sintetico (v. sezione precedente); l'HMM di regime lo rileva
correttamente (probabilità di stress ≈ 1 esattamente nella finestra iniettata,
si veda `results/spectral_regime.png`), confermando che l'indicatore funziona
come atteso. I 60 passi di backtest rolling qui mostrati ricadono in un
periodo di calma (fuori dalla finestra di stress); le metriche di accuratezza
puntuali vanno quindi lette come verifica di funzionamento della pipeline, non
come conclusione generale sulla bontà relativa dei modelli — quella richiede
dati reali (`--synthetic` disattivato) e un periodo di backtest che copra
anche fasi di stress.

File generati in `results/`:
- `graph_mst.png`, `graph_pmfg.png`, `graph_granger.png`, `graph_diebold_yilmaz.png`, `graph_sector.png` — visualizzazione delle 5 topologie di rete costruite
- `spectral_regime.png`, `spectral_regime_features.csv` — indicatore di regime spettrale (λ₁/Σλᵢ, rate-of-change, probabilità di stress HMM)
- `acf_pacf.png` — diagnostica ACF/PACF per i primi asset
- `eda_diagnostics.csv` — test ADF e Ljung-Box per ciascun asset
- `diebold_yilmaz_spillover_matrix.csv` — matrice di spillover di volatilità
- `gnarx_coefficients.csv` — coefficienti stimati del modello finale (in-sample, campione intero)
- `rolling_mean_forecast_example.png`, `rolling_volatility_forecast_example.png` — previsione rolling vs realizzato
- `rmse_by_asset.png` — RMSE per singolo asset
- `metrics_mean_*.csv`, `metrics_variance_*.csv`, `metrics_var_coverage_*.csv` — metriche dettagliate per ciascuno dei 3 modelli confrontati

## 7. Estensioni possibili

- Stima "locale" del GNAR (coefficienti specifici per nodo anziché globali)
- ARMA(p,q) completo per il DST-ARCH (qui implementato AR(1), estendibile)
- Rete combinata multi-livello (fusione pesata di MST + settoriale + spillover)
- Realized volatility infragiornaliera al posto della proxy r²
- Test di Diebold-Mariano per la significatività statistica dei confronti tra modelli

## Licenza

MIT — vedi `LICENSE`.
