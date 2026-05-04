---
layout: home
title: Home
nav_order: 1
description: >-
  Pipeline ML end-to-end per anomaly detection IoT con clustering KMeans/GMM,
  feature engineering temporale (rolling/diff/zscore), time-aware split senza
  leakage e soglia ottimale percentile su time-series industriali.
permalink: /
---

<div class="hero-banner" markdown="0">
  <h1>IoT Anomaly Detection &mdash; ML Pipeline</h1>
  <p>
    Dal CSV grezzo a <code>detect_anomalies()</code>: wrangling, feature
    engineering temporale, time&#8209;aware split, KMeans/GMM, soglia
    percentile, validazione vs ground&#8209;truth parziale. Riproducibile,
    modulare, deploy&#8209;ready.
  </p>
</div>

## In sintesi

Progetto di riferimento del percorso **Machine Learning Engineer** di
[DataMasters](https://datamasters.it/)/Skiller. Implementa l'intero flusso
di un sistema di **anomaly detection unsupervised** su time-series IoT
industriali, dal dato grezzo all'inferenza, con focus su **rigorosità
metodologica**, **prevenzione del leakage temporale** e **riproducibilità**.

L'anomalia è interpretata come una proprietà *relativa* al regime osservato:
i **pattern di funzionamento normale** vengono appresi con clustering sui
primi 7 giorni, poi sui 3 giorni successivi si flaggano le deviazioni
rispetto al contesto operativo. Le label parziali (`anomaly_label`,
`fault_code_true`) servono solo per validazione a posteriori.

<div class="kpi-grid" markdown="0">
  <div class="kpi-card">
    <div class="kpi-label">F1 holdout test</div>
    <div class="kpi-value">0.340</div>
    <div>3 giorni / 69 120 osservazioni</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">ROC-AUC test</div>
    <div class="kpi-value">0.879</div>
    <div>discriminazione score</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">PR-AUC test</div>
    <div class="kpi-value">0.315</div>
    <div>~13× sopra random (prevalenza 2.4%)</div>
  </div>
</div>

## Repository GitHub

- **Nome del repository**: `iot-anomaly-detection-clustering`
- **URL**: [github.com/fedcal/iot-anomaly-detection-clustering](https://github.com/fedcal/iot-anomaly-detection-clustering)
- **Documentazione (questo sito)**: pubblicata via **GitHub Pages** dalla cartella
  [`/docs`](https://github.com/fedcal/iot-anomaly-detection-clustering/tree/main/docs).

{: .note }
> La documentazione viene servita direttamente dai file Markdown della cartella
> `docs/`, processati da Jekyll con il tema **Just the Docs**.
> Ogni push su `main` aggiorna automaticamente il sito.

## Quick start

```bash
git clone https://github.com/fedcal/iot-anomaly-detection-clustering.git
cd iot-anomaly-detection-clustering
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

# Posiziona il dataset in data/raw/iot_synth_anomaly_clustering.csv
iot-detect              # pipeline completa ~60s
iot-detect --quick      # smoke test ~30s, K=5 fisso

# Soglia più aggressiva (più recall, meno precision)
iot-detect --threshold-percentile 95
```

Inferenza programmatica:

```python
from iot_anomaly.inference import detect_anomalies
import pandas as pd

df = pd.read_csv("nuove_misurazioni.csv", parse_dates=["timestamp"])

# Ritorna df + colonne anomaly_score, anomaly_pred
result = detect_anomalies(df)

# Solo le righe predette anomale
anomalies = detect_anomalies(df, return_only_anomalies=True)
```

## Risultati di riferimento (run quick, K=5)

| Set | Precision | Recall | F1 | ROC-AUC | PR-AUC | fault_code_recall |
|:--|--:|--:|--:|--:|--:|--:|
| Train | 0.300 | 0.156 | 0.205 | 0.820 | 0.186 | 0.173 |
| **Test** | **0.436** | **0.278** | **0.340** | **0.879** | **0.315** | **0.287** |

{: .tip }
> PR-AUC ≈ 0.32 con prevalenza 2.4% &rarr; **13&times; sopra il random**.
> Sul test set il modello cattura il 29% dei punti con `fault_code_true ≠ 0`,
> un ground-truth più affidabile di `anomaly_label`.

## Mappa della documentazione

### [Teoria](teoria/)

Fondamenti per leggere i risultati del progetto:

- [Anomaly detection unsupervised](teoria/01_anomaly_detection_unsupervised/) — point/contextual/collective anomaly, perché clustering.
- [Clustering KMeans & GMM](teoria/02_clustering_kmeans_gmm/) — algoritmi, scelta di K, silhouette/elbow/BIC.
- [Feature engineering temporale](teoria/03_feature_engineering_temporale/) — rolling, diff, z-score locale, calcolo per asset.
- [Soglia & metriche](teoria/04_soglia_e_metriche/) — percentile vs business-driven, ROC-AUC vs PR-AUC.
- [Split temporale & no leakage](teoria/05_split_temporale_no_leakage/) — time-aware split, anti-leakage, drift detection.

### [Scelte tecniche](scelte_tecniche/)

Decisioni architetturali e di modellazione:

- [Architettura](scelte_tecniche/architettura/) — moduli `src/`, flusso dati, CLI.
- [Scelte di modellazione](scelte_tecniche/scelte_modello/) — trade-off espliciti, K, soglia, finestra rolling.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.11 / 3.12 / 3.13 |
| ML | scikit-learn (KMeans, MiniBatchKMeans, GaussianMixture) |
| Data | pandas, numpy, scipy, pyarrow |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, ipykernel, nbformat |
| Persistenza | joblib |
| Documentazione | Jekyll + Just the Docs |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso
*Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti:
[**federicocalo.dev**](https://federicocalo.dev){: .btn .btn-purple }
