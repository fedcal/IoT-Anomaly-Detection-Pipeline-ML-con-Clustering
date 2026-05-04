---
layout: default
title: Teoria
nav_order: 2
has_children: true
permalink: /teoria/
description: >-
  Fondamenti teorici della pipeline IoT Anomaly Detection: anomaly detection
  unsupervised, clustering KMeans/GMM, feature engineering temporale, scelta
  della soglia e prevenzione del data leakage in time-series.
---

# Teoria

I cinque articoli di questa sezione costruiscono progressivamente le basi
necessarie per leggere il progetto **IoT Anomaly Detection** e per ragionare
in modo critico sui risultati del clustering su time-series industriali.

## Percorso consigliato di lettura

| Capitolo | Titolo | Concetti chiave |
|:--|:--|:--|
| 1 | [Anomaly detection unsupervised](01_anomaly_detection_unsupervised/) | Point/contextual/collective, supervised vs unsupervised, perché clustering |
| 2 | [Clustering KMeans & GMM](02_clustering_kmeans_gmm/) | KMeans, MiniBatchKMeans, GaussianMixture, scelta di K, silhouette/elbow/BIC |
| 3 | [Feature engineering temporale](03_feature_engineering_temporale/) | Rolling mean/std, derivate, z-score locale, calcolo per asset, PCA opzionale |
| 4 | [Soglia & metriche](04_soglia_e_metriche/) | Soglia percentile vs business-driven, precision/recall/F1, ROC-AUC vs PR-AUC |
| 5 | [Split temporale & no leakage](05_split_temporale_no_leakage/) | Time-aware split, anti-leakage in time-series, drift detection |

{: .note }
> Ogni capitolo è autocontenuto: leggi nell'ordine se vuoi una progressione
> didattica, oppure salta direttamente al capitolo che ti serve.

## Riferimenti trasversali

- Chandola, V., Banerjee, A. & Kumar, V. (2009) — *Anomaly Detection: A Survey*,
  ACM Computing Surveys 41(3).
- Aggarwal, C. C. (2017) — *Outlier Analysis*, Springer (2nd ed.).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009),
  capp. 14 (cluster analysis) e 8 (model assessment).
