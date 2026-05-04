---
sidebar_position: 1
title: Architettura del progetto
description: |
  Moduli, flusso dati IoT, CLI iot-detect.
---

# Architettura del progetto

## 1. Layout

```
iot-anomaly-detection-clustering/
├── README.md
├── LICENSE                       MIT — © 2026 Federico Calò
├── pyproject.toml                Build config + dipendenze
├── requirements.txt              Lock approssimativo
├── mkdocs.yml                    Config documentazione (MkDocs Material)
│
├── .github/
│   └── workflows/
│       └── docs.yml              Deploy automatico docs su GitHub Pages
│
├── src/iot_anomaly/              Libreria Python installabile
│   ├── __init__.py
│   ├── config.py                 Path, costanti, iperparametri
│   ├── data.py                   Load + time-aware split
│   ├── wrangling.py              Missing values per asset
│   ├── features.py               TimeSeriesFeatureEngineer (rolling, diff, zscore)
│   ├── clustering.py             KMeans/MiniBatch/GMM + select_k
│   ├── scoring.py                Distanza centroide + soglia percentile
│   ├── evaluation.py             Metriche + plot diagnostici
│   ├── inference.py              detect_anomalies()
│   └── pipeline.py               Orchestrator + CLI iot-detect
│
├── notebooks/                    Documentazione esecutiva
│   ├── 01_eda.ipynb
│   ├── 02_features_pipeline.ipynb
│   ├── 03_clustering_threshold.ipynb
│   └── 04_validation_inference.ipynb
│
├── docs/
│   ├── teoria/                   5 file Markdown didattici
│   └── scelte_tecniche/          Architettura, scelte di modello
│
├── data/raw/                     iot_synth_anomaly_clustering.csv (gitignored)
├── reports/                      Output (figures, models, metrics — gitignored)
│   ├── figures/
│   ├── models/                   *.joblib serializzati
│   └── metrics.json
│
├── scripts/
│   ├── build_notebooks.py
│   └── run_full.sh
└── tests/
```

## 2. Principi di design

Stesso pattern di [`ames-housing-price-pipeline`](https://github.com/fedcal/ames-housing-price-pipeline):

### 2.1 Codice in `src/`, narrativa nei notebook
La logica vive in `src/iot_anomaly/`. I notebook contengono solo importazioni, chiamate, e narrazione didattica. Modifiche al codice si riflettono automaticamente nei notebook (riavvia kernel).

### 2.2 Notebook generati da script
`scripts/build_notebooks.py` è la sorgente di verità. Diff Git puliti.

### 2.3 Pipeline as code (sklearn-compatible)
Il `TimeSeriesFeatureEngineer` è un `BaseEstimator + TransformerMixin`. Il modello finale (clustering + scoring + soglia) è incapsulato in un `AnomalyDetector`. L'inferenza ricarica un singolo `joblib`.

### 2.4 No leakage by construction
- Wrangling deterministico (ffill/bfill per asset) — può stare fuori dalla pipeline (non dipende da statistiche).
- Imputazione, scaling, soglia → **sempre** calcolate sul solo training.
- Rolling back-looking, per asset (`groupby('asset_id')`).
- Time-aware split, mai casuale.

## 3. Flusso di esecuzione

### 3.1 Training completo

```bash
iot-detect              # full run, ~60s
iot-detect --quick      # K=5 fisso, ~30s
iot-detect --use-pca    # con PCA prima del clustering
iot-detect --threshold-percentile 95  # soglia più aggressiva
```

Sequenza interna:

1. `load_raw()` → DataFrame ordinato per (asset, time).
2. `add_missing_flags` + `fill_missing_per_asset` → wrangling.
3. `TimeSeriesFeatureEngineer` → rolling, diff, zscore.
4. `time_split` → 7 giorni train / 3 giorni test.
5. `StandardScaler` + (opzionale) `PCA`.
6. `select_k_by_silhouette` su K ∈ \{3..10\}.
7. `fit_minibatch_kmeans` + `fit_anomaly_detector` (soglia p99).
8. `evaluate` su train e test vs `anomaly_label` e `fault_code_true`.
9. Salvataggio joblib + metriche JSON + figure.

### 3.2 Inferenza

```python
from iot_anomaly.inference import detect_anomalies

import pandas as pd
df_new = pd.read_csv("nuovi_dati.csv", parse_dates=["timestamp"])
result = detect_anomalies(df_new)
print(result[result.anomaly_pred == 1])
```

`detect_anomalies()` riapplica wrangling, FE, scaling, scoring usando gli artefatti serializzati. Output: DataFrame originale + colonne `anomaly_score`, `anomaly_pred`.

## 4. Riproducibilità

- **Stesso dataset**: la pipeline carica solo `data/raw/iot_synth_anomaly_clustering.csv`.
- **Stesse versioni**: `requirements.txt` pinnato a range minor.
- **Stesso seed**: `RANDOM_STATE=42` propagato a `MiniBatchKMeans`, `silhouette_score` (via sample), `KMeans`, `GMM`.

Con stesso ambiente, esecuzioni successive di `iot-detect` producono metriche bit-identiche.

## 5. Trade-off espliciti

| Decisione | Vantaggio | Costo |
|---|---|---|
| Time-aware split rigido (7+3) | Realismo | Solo 1 holdout, non K-fold temporale |
| MiniBatchKMeans default | 10× più veloce di KMeans full | Inertia ~1-3% peggiore |
| Soglia p99 fissa | Configurabile, interpretabile | Non ottimale per ogni asset |
| Niente label nel training | Generalizzazione a anomalie nuove | Performance inferiori vs supervised |
| FE temporale solo rolling+diff+zscore | Interpretabile, veloce | Nessuna feature multivariata (es. cross-correlations) |

## 6. Estensioni naturali

1. **GMM con BIC** in alternativa a KMeans+silhouette.
2. **DBSCAN/HDBSCAN** per cluster non sferici e detection di noise nativo.
3. **Soglia per asset**: ogni asset ha la sua distribuzione; un percentile globale è sub-ottimale.
4. **Feature multivariate**: cross-correlation fra sensori, FFT su finestre brevi.
5. **Modelli sequenziali** (LSTM autoencoder) per collective anomalies.
6. **Drift detection** (KS-test) + retraining schedulato.
7. **API REST** (FastAPI) che espone `detect_anomalies()`.
