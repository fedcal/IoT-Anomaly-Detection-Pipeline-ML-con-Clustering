---
sidebar_position: 1
title: Anomaly detection unsupervised
description: |
  Approcci unsupervised, distance-based, density-based, isolation forest.
---

# Anomaly detection non supervisionata

## 1. Cos'è un'anomalia

Un'**anomalia** è un'osservazione che si discosta significativamente dal comportamento atteso. Il problema è che "atteso" è soggettivo: dipende dal **contesto operativo** e dal **periodo** osservato.

In manutenzione predittiva industriale, lavoriamo tipicamente con tre famiglie di anomalie:

1. **Point anomaly**: un singolo campione fuori dal range normale (es. picco di temperatura).
2. **Contextual anomaly**: un valore "normale in assoluto" ma anomalo nel contesto attuale (es. RPM=3000 è normale a pieno carico, anomalo a vuoto).
3. **Collective anomaly**: un pattern di valori che, presi singolarmente, sono normali, ma in sequenza non lo sono (es. 30 minuti di temperatura in lieve crescita verso il limite).

Il dataset Ames-IoT del project work contiene anomalie di tutti e tre i tipi. La pipeline che costruiamo cattura bene point e contextual; le collective richiedono modelli sequenziali (LSTM/transformer) fuori scope.

## 2. Supervised vs unsupervised

| Approccio | Quando usarlo | Limiti |
|---|---|---|
| **Supervised** (classificazione binaria con label) | Si hanno label complete e bilanciate | Richiede label per OGNI tipo di anomalia (lavoro umano costoso); non rileva anomalie nuove. |
| **Unsupervised** (no label durante il training) | Si vuole rilevare anomalie *non viste prima*; le label sono parziali o assenti | Performance generalmente inferiori al supervised quando le label sono complete; sensibile alle assunzioni sul "comportamento normale". |
| **Semi-supervised** (training su solo dati normali) | Si hanno label di "normale" certe ma poche o nessuna di anomalia | Sotto-caso dell'unsupervised se assumi che il training sia normale. |

In Ames-IoT abbiamo **label parziali**: solo ~2.4% dei campioni è etichettato come anomalia, e la documentazione dichiara esplicitamente che le label sono "non necessariamente complete". Quindi l'approccio canonico è **unsupervised + validazione debole** sulle label disponibili.

**Regola fondamentale**: le label NON entrano nel training del clustering. Servono solo per la validazione a posteriori.

## 3. Perché il clustering è una scelta naturale

Il clustering presume che il "comportamento normale" si manifesti come un piccolo numero di **stati operativi ricorrenti** (regimi). Una macchina industriale non è in uno stato continuo: alterna idle, accelerazione, regime, decelerazione, idle. Ogni stato ha un proprio "fingerprint" di sensori.

Il clustering apprende questi fingerprint dai dati. Un punto è anomalo quando **non assomiglia a nessuno** dei fingerprint normali — cioè è lontano da tutti i centroidi.

Vantaggi:
- Nessuna label richiesta.
- Cluster interpretabili (uno per regime operativo).
- Soglia configurabile (precision vs recall).
- Latenza inferenza bassa: O(K × d) per punto, K = numero cluster, d = dimensione feature.

Limiti:
- Funziona se le anomalie sono **lontane** dai centroidi normali. Anomalie che cadono *dentro* un cluster (lo "shift" lento di un cuscinetto che progressivamente cambia firma di vibrazione) non vengono rilevate.
- Sensibile alla **scelta delle feature**. Senza feature engineering temporale (rolling, diff, zscore) il clustering vede solo lo stato istantaneo, non la dinamica.
- Sensibile alla **scelta di K**. K piccoli sotto-discriminano (mescolano regimi); K grandi sovra-discriminano (anomalie diventano cluster a sé).

## 4. Anomalia "relativa al contesto"

Un'enfasi importante del PW: l'anomalia è una proprietà **relativa al periodo osservato**. Se i primi 7 giorni vedono solo macchine al 50% di carico, e poi nei test 3 giorni il carico sale al 90%, il modello potrebbe segnalare il 90% come anomalo — quando in realtà è solo un nuovo regime non visto durante il training.

Questo è un **falso positivo per drift di distribuzione** (concept drift), non un vero guasto. Lo si gestisce in tre modi:

1. **Espandere il training** a includere più storicità.
2. **Drift detection** automatica (KS-test mensile sui sensori): quando il drift è statistico significativo, retrainare il modello.
3. **Soglia per regime**: una soglia diversa per ogni stato `regime` osservato.

La nostra pipeline fa la base (1); le estensioni (2-3) sono documentate come roadmap.

## 5. Riferimenti

- **Aggarwal, C.** (2017), *Outlier Analysis*, Springer.
- **Chandola et al.** (2009), *Anomaly Detection: A Survey*, ACM Computing Surveys 41(3).
- **Goldstein & Uchida** (2016), *A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms*, PLoS ONE.
