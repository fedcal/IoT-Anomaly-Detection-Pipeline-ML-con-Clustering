---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline ML per il rilevamento di anomalie in dati IoT tramite clustering, feature engineering temporale e ottimizzazione della soglia.
slug: /intro
---

# IoT Anomaly Detection — ML Pipeline

Pipeline ML didattica e **production-friendly** che, a partire da time-series IoT industriali, costruisce feature aggregate temporali, addestra clustering KMeans/GMM, definisce una soglia su percentile delle distanze e segnala anomalie in modo riproducibile: wrangling → feature engineering temporale → time-aware split → KMeans/MiniBatch/GMM → soglia su percentile → `detect_anomalies()`.

:::tip In una riga
*Da time-series IoT grezza a `detect_anomalies()` con feature engineering, clustering e soglia su percentile.*
:::

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/IoT-Anomaly-Detection-Pipeline-ML-con-Clustering`](https://github.com/fedcal/IoT-Anomaly-Detection-Pipeline-ML-con-Clustering) |
| Documentazione | [https://fedcal.github.io/IoT-Anomaly-Detection-Pipeline-ML-con-Clustering/](https://fedcal.github.io/IoT-Anomaly-Detection-Pipeline-ML-con-Clustering/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

## Mappa della documentazione

### [Teoria](/docs/category/teoria)

1. [Anomaly detection unsupervised](./teoria/01-anomaly-detection-unsupervised.md) — Approcci unsupervised, distance-based, density-based, isolation forest.
2. [Clustering: KMeans & GMM](./teoria/02-clustering-kmeans-gmm.md) — Differenze KMeans vs GMM, quando usarli per anomaly detection.
3. [Feature engineering temporale](./teoria/03-feature-engineering-temporale.md) — Aggregati su finestre, rolling stats, lag features.
4. [Soglia & metriche](./teoria/04-soglia-metriche.md) — Percentile-based threshold, F1, ROC-AUC, PR-AUC su classi sbilanciate.
5. [Split temporale & no leakage](./teoria/05-split-temporale-no-leakage.md) — Time-aware split, prevenzione del leakage in time series.

### [Scelte tecniche](/docs/category/scelte-tecniche)

- [Architettura del progetto](./scelte-tecniche/architettura.md) — Moduli, flusso dati IoT, CLI iot-detect.
- [Scelte di modellazione: razionale](./scelte-tecniche/scelte-modello.md) — Razionale KMeans + GMM, scelta soglia, gestione drift.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
