---
title: IoT Anomaly Detection — Pipeline ML
description: >-
  Pipeline end-to-end di Machine Learning per il rilevamento di anomalie su
  time-series IoT industriali tramite clustering KMeans/GMM, time-aware split
  e validazione su ground-truth parziale.
---

<div class="hero-banner">
  <h1>IoT Anomaly Detection — Pipeline ML con Clustering</h1>
  <p>
    Dal dato grezzo a <code>detect_anomalies()</code>: time-aware split,
    feature engineering rolling, KMeans/GMM, soglia su percentile, validazione
    vs ground-truth parziale. Riproducibile, modulare, deploy-ready.
  </p>
</div>

## Il progetto in 30 secondi

Il **Project Work 2** del percorso *Machine Learning Engineer* (DataMasters) chiede di costruire una pipeline ML capace di:

1. caricare un dataset IoT industriale (16 asset × 10 giorni × 1 minuto = 230 400 osservazioni),
2. apprendere i **pattern di funzionamento normale** osservati nello storico,
3. individuare **deviazioni significative** rispetto al contesto operativo.

L'anomalia è interpretata come una proprietà *relativa* al regime osservato: NON una condizione assoluta del sistema. Coerentemente, le `anomaly_label` non vengono usate per il training del clustering — solo per validazione a posteriori.

## Repository GitHub raccomandato

> **Nome del repository pubblico**: `iot-anomaly-detection-clustering`
> URL atteso: <https://github.com/fedcal/iot-anomaly-detection-clustering>

## Quick start

```bash
git clone https://github.com/fedcal/iot-anomaly-detection-clustering.git
cd iot-anomaly-detection-clustering

python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

# Pipeline completa (~60s) con dataset in data/raw/
iot-detect

# Smoke test (~30s)
iot-detect --quick
```

Inferenza:

```python
from iot_anomaly.inference import detect_anomalies
import pandas as pd

df = pd.read_csv("nuovi_dati.csv", parse_dates=["timestamp"])
result = detect_anomalies(df)
print(result[result.anomaly_pred == 1])
```

## Risultati di riferimento (run quick, K=5)

| Set | Precision | Recall | F1 | ROC-AUC | PR-AUC | fault_code_recall |
|---|---|---|---|---|---|---|
| Train | 0.300 | 0.156 | 0.205 | 0.820 | 0.186 | 0.173 |
| **Test** | **0.436** | **0.278** | **0.340** | **0.879** | **0.315** | **0.287** |

PR-AUC ≈ 0.32 con prevalenza 2.4% → **13× sopra il random**. Sul test set il modello cattura il 29 % dei punti con `fault_code_true ≠ 0` (un ground-truth più affidabile di `anomaly_label`).

## Mappa della documentazione

- **[Teoria](teoria/01_anomaly_detection_unsupervised.md)** — i fondamenti: cos'è un'anomalia, perché clustering, KMeans vs GMM, feature engineering temporale, scelta della soglia, prevenzione del leakage.
- **[Scelte tecniche](scelte_tecniche/architettura.md)** — architettura del pacchetto, decisioni di modellazione, trade-off espliciti, estensioni naturali.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di DataMasters/Skiller.

Per altri progetti e contatti: [federicocalo.dev](https://federicocalo.dev).
