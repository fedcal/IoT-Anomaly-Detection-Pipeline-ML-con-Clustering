---
sidebar_position: 5
title: Split temporale & no leakage
description: |
  Time-aware split, prevenzione del leakage in time series.
---

# Split temporale e prevenzione del leakage in time-series

## 1. Lo split casuale è SEMPRE sbagliato su time-series

Su dati tabulari iid (es. Ames Housing), uno split casuale (`train_test_split(shuffle=True)`) è la pratica standard. Su dati con dimensione temporale è **sempre sbagliato**.

Esempio: un modello che predice il prezzo di una casa il 15 gennaio 2010 vede in training case vendute il 20 marzo 2010. Le condizioni di mercato di marzo 2010 sono già "il futuro" rispetto al 15 gennaio. La performance misurata in CV sovrastima quella reale in deployment.

In anomaly detection IoT è ancora peggio: un punto anomalo isolato in un asset il 9 febbraio non è informativo se nel training il modello ha visto i campioni successivi del 9 febbraio. Il modello impara a "interpolare" i punti vicini nel tempo.

## 2. Il time-aware split corretto

Definisce un **cutoff temporale** $t^*$:

- **Training**: tutti i punti con timestamp $< t^*$.
- **Test**: tutti i punti con timestamp $\geq t^*$.

Tutti i 16 asset sono presenti in entrambi i set; cambia solo la finestra temporale di osservazione.

Per il PW IoT, $t^* = 2025\text{-}02\text{-}08$ (cutoff dopo 7 giorni). Il training rappresenta i primi 7 giorni di operatività ("comportamento normale" assunto), il test gli ultimi 3 giorni (in cui il modello deve generalizzare al futuro non visto).

## 3. Discussione dell'assunzione "training = normale"

Il PW richiede di motivare e discutere l'assunzione. Tre scenari:

### 3.1 Caso ottimistico: training davvero normale
Se in produzione facciamo retraining periodico (ogni mese, settimana, etc.) e i primi giorni post-retrain sono in stato regolare, l'assunzione regge. Il modello impara i fingerprint dei regimi normali e segnala come anomalia tutto ciò che si discosta.

### 3.2 Caso intermedio: training contiene poche anomalie isolate
Le anomalie nei primi 7 giorni del nostro dataset (~2.4%) "diluiscono" la stima della normalità. Il KMeans dovrebbe collocarle ai bordi dei cluster ("rumore tollerato"). Il p99 della distribuzione di distanze le include comunque tra i punti normali → soglia leggermente sovra-stimata.

Effetto pratico: il modello trovato è leggermente meno sensibile (più precision, meno recall) di quanto sarebbe con training perfettamente normale.

### 3.3 Caso pessimistico: training in regime guasto
Se il training fosse stato preso durante un guasto cronico, il modello apprenderebbe il comportamento guasto come "normale" e classificherebbe come anomali i ritorni alla normalità. Disastroso.

Questo scenario si previene con:
- **Validazione manuale del training set** prima di lanciare il training.
- **Cross-check** con label parziali: se il training ha > 5% di `fault_code_true != 0`, alert.
- **Drift detection** continua sulle distribuzioni dei sensori.

## 4. Cross-validation temporale

Per dataset abbastanza grandi, si fa CV temporale (TimeSeriesSplit di sklearn):

```
Fold 1: train [0..30%]   test [30%..40%]
Fold 2: train [0..40%]   test [40%..50%]
Fold 3: train [0..50%]   test [50%..60%]
...
```

Ogni fold espande il training nel passato e valuta sul futuro immediato. Mantiene l'ordine temporale e simula il deployment realistico.

In Ames-IoT (10 giorni) abbiamo poco margine: fare 5 fold di TimeSeriesSplit lascia ogni fold con 1-2 giorni di test. Per il PW didattico ci limitiamo a un singolo holdout 7+3, ma la generalizzazione a TimeSeriesSplit è banale.

## 5. Drift detection

Anche il time-aware split non protegge dal **distribution drift**: se a febbraio cambia il regime di carico (più clienti, condizioni ambientali diverse), il training del 1-7 febbraio non è più rappresentativo.

Indicatori di drift:
- **Kolmogorov-Smirnov test** sui sensori chiave: confronta distribuzione train vs ultimi N minuti.
- **Population Stability Index (PSI)**: misura quanto la distribuzione di una feature è cambiata.
- **Anomaly rate**: se la frazione di punti predetti anomali sale stabilmente sopra il p99 atteso, c'è drift.

Quando il drift è significativo, il modello va **retreinato** sui dati recenti.

## 6. Riepilogo: regole d'oro

1. **Split temporale, mai random**, su dati con timestamp.
2. **Tutte le statistiche** (imputer, scaler, encoder, soglia) calcolate solo sul training.
3. **Rolling/feature** computate per asset, back-looking (no `center=True`).
4. **CV nested** se si fa hyperparameter tuning: split temporale fuori, normale dentro.
5. **Holdout temporale finale** per la stima della generalizzazione.
6. **Drift detection in produzione** + **retraining** schedulato.

## 7. Riferimenti

- **Bergmeir & Benítez** (2012), *On the Use of Cross-Validation for Time Series Predictor Evaluation*, Information Sciences.
- **Hyndman & Athanasopoulos** (2021), *Forecasting: Principles and Practice*, cap. 5.
- **sklearn**: [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
