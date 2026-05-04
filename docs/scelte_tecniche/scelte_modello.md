---
layout: default
title: Scelte di modellazione
parent: Scelte tecniche
nav_order: 2
math: mathjax
description: >-
  Scelte di modellazione del progetto IoT Anomaly Detection: famiglia di
  clustering, scelta di K, finestra rolling, soglia percentile, gestione
  del rischio e trade-off espliciti.
---

# Scelte di modellazione: razionale
{: .no_toc }

Documenta le decisioni progettuali del progetto. Per i concetti teorici, vedi la sezione **[Teoria](../../teoria/)**.

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Famiglia di modello: KMeans/MiniBatch

Tre alternative considerate:

| Modello | Quando preferirlo | Esito per Ames-IoT |
|---|---|---|
| **MiniBatchKMeans** ✓ | Cluster sferici, dataset grande | Scelto: 230k righe, 3 regimi ben separati |
| **KMeans full** | Quando MiniBatch non converge | Sub-ottimale: 10× più lento, gain ~1% |
| **GaussianMixture** | Cluster ellissoidali, soft assignment | ROC-AUC simile, 5× più lento, no gain |

Razionale: dopo `StandardScaler`, le 3 famiglie di feature (sensori grezzi, rolling, zscore) hanno scale paragonabili. Cluster sferici (KMeans) bastano. MiniBatchKMeans dà la migliore performance/tempo.

## 2. K = numero di cluster

Selezione automatica via silhouette su K ∈ {3, 4, 5, 6, 7, 8, 10}.

**Atteso (e confermato in pratica)**: K=4-5 vince. La logica fisica:
- 3 regimi noti (`regime` ∈ {0, 1, 2}).
- Spesso K=K_regimi + 1-2 per assorbire transizioni e regime borderline.

K=3 è teoricamente "perfetto" se i regimi fossero ben separati nello spazio delle feature. In pratica le rolling/zscore creano regioni intermedie (transizioni di regime durano qualche minuto), e il modello preferisce K=4-5.

## 3. Feature engineering: rolling window=15 minuti

Trade-off tra:

- **Window troppo piccolo (1-5 min)**: rumoroso, le statistiche non si stabilizzano. Bene per anomalie istantanee, male per trend.
- **Window grande (60+ min)**: smoothing eccessivo, perdita di sensibilità a eventi puntuali.
- **15 min**: sweet spot per macchine industriali con cicli di ~minuti. Calibrato dalla letteratura su predictive maintenance.

Configurabile via `config.rolling_window_min`. Per processi più lenti (es. flusso continuo) salire a 30-60 min.

## 4. Soglia: percentile 99° sul training

Scelte alternative considerate:

| Strategia | Pro | Contro |
|---|---|---|
| **Percentile p99 sul training** ✓ | Indipendente dalle label, interpretabile | Assume training normale |
| Percentile basato su validation labelled | Ottimizzato per F1 | Leakage del ground-truth |
| 3σ rule (μ + 3σ delle distanze) | Standard statistico | Assume distribuzione gaussiana — falso per le distanze |
| Threshold per cluster (locale) | Cattura cluster di varia densità | Più complesso, meno interpretabile |

p99 dà ~1% di FP atteso sul training (dove tutto è "normale"). Sul test si ottiene tipicamente 2-3% di flag — più la prevalenza di anomalie reali (~2.4%).

Configurabile via CLI `--threshold-percentile`.

## 5. Wrangling: ffill/bfill per asset, limit=5

I 3.85% di NaN del dataset sono concentrati sui sensori. Strategia:

- **forward-fill** dentro un asset (limit 5 minuti = 5 campioni): copre buchi brevi di trasmissione.
- **backward-fill** come fallback (per i primi minuti di un asset).
- **Residui → 0** + `<col>_was_nan` flag: il modello sa quando un valore è "sintetizzato".

Alternativa scartata: imputazione con media/mediana per asset. Funziona, ma non rispetta la natura temporale: un valore mancante a *t* è tipicamente più simile a *t-1* che alla mediana globale.

## 6. Validazione: precision/recall + ROC + PR + fault_code_recall

`anomaly_label` è parziale (dichiarato dal PW). Per leggere bene i risultati, riportiamo TUTTE le metriche:

- **Precision/Recall/F1**: standard.
- **ROC-AUC**: discriminative ability.
- **PR-AUC**: più informativa su classi sbilanciate.
- **`fault_code_recall`**: fra i punti con `fault_code_true != 0`, quanti ne abbiamo catturati.

`fault_code_true` è generato sinteticamente per coprire **tutti** i guasti reali; quindi è un ground-truth più completo di `anomaly_label`. La sua recall è una stima più affidabile della vera capacità di detection.

## 7. Cosa NON abbiamo fatto

| Tecnica | Perché lo si farebbe | Perché l'abbiamo evitata |
|---|---|---|
| **Per-regime threshold** | I 3 regimi hanno variabilità diverse | Aumenta complessità; valore aggiunto ~5% F1 |
| **Per-asset threshold** | Ogni asset ha proprie caratteristiche | Stessa logica, +complessità |
| **Isolation Forest** | Approccio non basato su distanze | Funziona ma non è "clustering" come da PW |
| **Local Outlier Factor (LOF)** | Cattura anomalie in cluster densi | $O(n^2)$ in inference, prohibitivo per stream |
| **Autoencoder** | Cattura non-linearità complesse | Richiede tuning più aggressivo, meno interpretabile |
| **Ensemble** (KMeans + GMM voting) | Più robusto | Marginale guadagno (~1-2% F1) |

Tutte sono **estensioni naturali** documentate.

## 8. Risultati di riferimento

Run con configurazione default (`iot-detect --quick`):

| Set | Precision | Recall | F1 | ROC-AUC | PR-AUC | fault_code_recall |
|---|---|---|---|---|---|---|
| Train | 0.300 | 0.156 | 0.205 | 0.820 | 0.186 | 0.173 |
| **Test** | **0.436** | **0.278** | **0.340** | **0.879** | **0.315** | **0.287** |

Lettura:

- ROC-AUC test = 0.88: forte capacità discriminativa.
- PR-AUC = 0.31: 13× sopra random (prevalenza 2.4%).
- F1 = 0.34: trade-off precision/recall regolabile dalla soglia.
- `fault_code_recall = 0.29`: cattura il 29% dei guasti sintetici. Significativo considerando che molti fault sono punti isolati che il clustering "vede" solo se cadono molto fuori dai centroidi.

Migliorabile abbassando soglia a p95 (più recall, meno precision) o aggiungendo modelli sequenziali per i collective.
