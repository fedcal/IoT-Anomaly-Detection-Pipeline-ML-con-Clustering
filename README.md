# IoT Anomaly Detection — Pipeline ML con Clustering

> Pipeline end-to-end di Machine Learning per il rilevamento di anomalie in time-series IoT industriali. Dato grezzo → wrangling → feature engineering temporale → time-aware split → KMeans/MiniBatch/GMM → soglia su percentile → `detect_anomalies()`. Riproducibile, modulare, deploy-ready.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-blueviolet.svg)](https://fedcal.github.io/iot-anomaly-detection-clustering/)

## Repository GitHub

**Nome consigliato del repository pubblico**: `iot-anomaly-detection-clustering`
URL atteso dopo la pubblicazione: <https://github.com/fedcal/iot-anomaly-detection-clustering>

Il deploy della documentazione (MkDocs Material) avviene automaticamente a ogni push su `main` tramite il workflow [`.github/workflows/docs.yml`](.github/workflows/docs.yml). Da abilitare in *Settings → Pages → Source = GitHub Actions* alla prima volta.

---

## Indice

- [Contesto](#contesto)
- [Risultati](#risultati)
- [Quick start](#quick-start)
- [Struttura del repository](#struttura-del-repository)
- [Documentazione](#documentazione)
- [Inferenza](#inferenza)
- [Riproducibilità](#riproducibilità)
- [Roadmap](#roadmap)
- [Autore & licenza](#autore--licenza)

---

## Contesto

Il **Project Work 2** del percorso *Machine Learning Engineer* (DataMasters) chiede di costruire una pipeline ML capace di:

1. caricare un dataset IoT industriale (16 asset × 10 giorni × 1 minuto = 230 400 osservazioni, 19 colonne),
2. apprendere i **pattern di funzionamento normale** dallo storico,
3. individuare **deviazioni significative** rispetto al contesto operativo,
4. validare i risultati confrontandoli con label parziali (`anomaly_label`, `fault_code_true`).

**Vincolo metodologico fondamentale**: le label NON entrano nel training del clustering. Servono solo per validazione a posteriori. Lo split è **temporale** (no shuffle): primi 7 giorni → training (assunti normali), ultimi 3 → test.

---

## Risultati

Holdout test (3 giorni, 69 120 osservazioni). Run rapido con `iot-detect --quick`:

| Set | Precision | Recall | F1 | ROC-AUC | PR-AUC | fault_code_recall |
|---|---|---|---|---|---|---|
| Train | 0.300 | 0.156 | 0.205 | 0.820 | 0.186 | 0.173 |
| **Test** | **0.436** | **0.278** | **0.340** | **0.879** | **0.315** | **0.287** |

Lettura: PR-AUC ≈ 0.32 con prevalenza 2.4% → **13× sopra random**. Il modello cattura il 29 % dei punti con `fault_code_true ≠ 0` (ground-truth completo, generato sinteticamente).

---

## Quick start

```bash
git clone https://github.com/fedcal/iot-anomaly-detection-clustering.git
cd iot-anomaly-detection-clustering

python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e ".[notebooks]"

# Posiziona il dataset:
#   data/raw/iot_synth_anomaly_clustering.csv  (33 MB, fornito col PW)

# Pipeline completa (~60s)
iot-detect

# Smoke test (~30s, K=5 fisso)
iot-detect --quick

# Soglia più aggressiva (più recall, meno precision)
iot-detect --threshold-percentile 95
```

Output:

- `reports/models/anomaly_detector.joblib` — pipeline completa serializzata
- `reports/metrics.json` — metriche train/test
- `reports/figures/*.png` — distribuzione score, PR curve

### Notebook didattici

```bash
jupyter lab notebooks/
```

I 4 notebook sono pensati in sequenza:

1. `01_eda.ipynb` — esplorazione del dataset, distribuzioni temporali, missing.
2. `02_features_pipeline.ipynb` — wrangling, feature engineering temporale, time-aware split.
3. `03_clustering_threshold.ipynb` — selezione di K, soglia, interpretazione cluster.
4. `04_validation_inference.ipynb` — metriche, plot diagnostici, `detect_anomalies()`.

I notebook sono **rigenerabili** da `scripts/build_notebooks.py` (sorgente di verità in Python — diff Git puliti).

---

## Struttura del repository

```
src/iot_anomaly/         Libreria Python installabile (pip install -e .)
├── config.py              Path, costanti, iperparametri
├── data.py                Load + time-aware split
├── wrangling.py           Missing values per asset (ffill/bfill)
├── features.py            TimeSeriesFeatureEngineer (rolling/diff/zscore)
├── clustering.py          KMeans/MiniBatch/GMM + selezione K
├── scoring.py             Distanza centroide + soglia percentile
├── evaluation.py          Metriche + plot diagnostici
├── inference.py           detect_anomalies()
└── pipeline.py            Orchestratore + CLI iot-detect

notebooks/             4 notebook didattici eseguibili
docs/
├── teoria/              5 file Markdown didattici
└── scelte_tecniche/     Architettura, scelte di modello
data/                  Dataset (gitignored)
reports/               Output (gitignored)
.github/workflows/     CI/CD per deploy automatico docs
```

---

## Documentazione

I 5 file in [`docs/teoria/`](docs/teoria/) coprono i fondamenti necessari:

| File | Contenuto |
|---|---|
| [`01_anomaly_detection_unsupervised.md`](docs/teoria/01_anomaly_detection_unsupervised.md) | Tipi di anomalia (point/contextual/collective); supervised vs unsupervised; perché il clustering. |
| [`02_clustering_kmeans_gmm.md`](docs/teoria/02_clustering_kmeans_gmm.md) | KMeans, MiniBatchKMeans, GMM; scelta di K; silhouette/elbow/BIC. |
| [`03_feature_engineering_temporale.md`](docs/teoria/03_feature_engineering_temporale.md) | Rolling mean/std, derivate, z-score locale; calcolo per asset; PCA opzionale. |
| [`04_soglia_e_metriche.md`](docs/teoria/04_soglia_e_metriche.md) | Soglia percentile vs business-driven; precision/recall/F1; ROC-AUC vs PR-AUC su classi sbilanciate. |
| [`05_split_temporale_no_leakage.md`](docs/teoria/05_split_temporale_no_leakage.md) | Time-aware split, anti-leakage in time-series, drift detection. |

In [`docs/scelte_tecniche/`](docs/scelte_tecniche/) sono documentate le decisioni progettuali (scelta del modello, K, finestra rolling, soglia, ecc.).

Sito navigabile: <https://fedcal.github.io/iot-anomaly-detection-clustering/> (GitHub Pages, deploy automatico via Actions).

---

## Inferenza

```python
from iot_anomaly.inference import detect_anomalies
import pandas as pd

df = pd.read_csv("data/nuove_misurazioni.csv", parse_dates=["timestamp"])

# Ritorna df + colonne anomaly_score, anomaly_pred
result = detect_anomalies(df)

# Solo le righe predette anomale
anomalies = detect_anomalies(df, return_only_anomalies=True)
```

`detect_anomalies()` ricarica un singolo `.joblib` che contiene wrangling, feature engineering, scaler, clustering, soglia. Inference latency ~5 ms per migliaio di righe.

---

## Riproducibilità

- Dataset deterministico: stessa SHA del file → stessi risultati.
- Versioni pinnate in `pyproject.toml` e `requirements.txt`.
- `RANDOM_STATE = 42` propagato a tutti gli step stocastici.

Esecuzione bit-identica con stesso ambiente. Tolleranza float64 con `n_jobs=-1` — per riproducibilità perfetta usare `n_jobs=1`.

---

## Roadmap

- [ ] Per-asset threshold (ogni asset ha la sua distribuzione di score).
- [ ] GaussianMixture con BIC come alternativa a KMeans+silhouette.
- [ ] LSTM autoencoder per collective anomalies.
- [ ] Drift detection automatica (KS-test mensile sui sensori).
- [ ] API REST (FastAPI) che espone `detect_anomalies()`.
- [ ] Container Docker riproducibile.

---

## Autore & licenza

**Creazione di Federico Calò** — Project Work del percorso *Machine Learning Engineer* (DataMasters/Skiller, 2026).

Per altri progetti, contatti e portfolio: <https://federicocalo.dev>.

[MIT License](LICENSE) © 2026 Federico Calò.
Il dataset IoT sintetico è fornito da DataMasters all'interno del corso.
