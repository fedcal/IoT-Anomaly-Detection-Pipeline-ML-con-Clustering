---
layout: default
title: Clustering KMeans & GMM
parent: Teoria
nav_order: 2
math: mathjax
description: >-
  Algoritmi di clustering per anomaly detection unsupervised: KMeans,
  MiniBatchKMeans, GaussianMixture; scelta di K con silhouette, elbow e BIC;
  trade-off computazionali su dataset IoT.
---

# Clustering: KMeans, MiniBatchKMeans, GaussianMixture
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. KMeans — l'algoritmo di base

KMeans partiziona i punti in $K$ cluster minimizzando la **somma dei quadrati delle distanze** dai centroidi:

$$
\mathcal{L}(\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K) = \sum_{i=1}^{n} \min_{k \in \{1,\ldots,K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

L'algoritmo (Lloyd's):

1. Inizializza $K$ centroidi (k-means++ è lo standard).
2. **Assignment**: per ogni punto, assegnalo al centroide più vicino.
3. **Update**: ricalcola ogni centroide come media dei punti assegnati.
4. Ripeti fino a convergenza (centroidi stabili o max iterazioni).

Complessità per iterazione: $O(n \cdot K \cdot d)$ con $n$ punti, $K$ cluster, $d$ feature.

### 1.1 Limiti del KMeans
- **Cluster sferici**: minimizza distanze euclidee → assume che i cluster abbiano stessa varianza in tutte le direzioni.
- **Sensibile alla scala**: una feature in metri (~1000) domina su una in (0,1). **Scaling obbligatorio** prima di KMeans.
- **K va scelto a priori**: nessun metodo "esatto"; si usa elbow + silhouette.
- **Scaling sui valori, non sulla forma**: due cluster ovali allungati si sovrappongono male in KMeans.

## 2. MiniBatchKMeans — KMeans efficiente

Per dataset > 100k righe, KMeans full batch diventa lento. **MiniBatchKMeans** (Sculley, 2010) processa il dataset in mini-batch (es. 4096 punti per batch), aggiornando i centroidi incrementalmente.

Risultato pratico:
- 10-100× più veloce di KMeans full.
- Inertia leggermente peggiore (~1-3%).
- Stessa complessità asintotica, ma costanti drasticamente migliori.

Sul nostro dataset (230k righe × 43 feature), KMeans full batch impiega ~30s per K=5; MiniBatchKMeans ~2s. Per il tuning di K (testiamo 7 valori) la differenza è 3 minuti vs 15 secondi.

## 3. Gaussian Mixture Model (GMM)

GMM è una generalizzazione probabilistica di KMeans. Modella ogni cluster come una **gaussiana multivariata** con propria media $\boldsymbol{\mu}_k$ e covarianza $\boldsymbol{\Sigma}_k$:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

con $\pi_k$ prior del cluster. Si addestra con l'algoritmo **EM (Expectation-Maximization)**.

### 3.1 Vantaggi
- **Cluster ellissoidali**: $\boldsymbol{\Sigma}_k$ cattura correlazioni tra feature.
- **Soft assignment**: ogni punto ha una probabilità di appartenere a ognuno dei $K$ cluster, non un'etichetta hard.
- **Anomaly score nativo**: `-log p(x)` è una "distanza probabilistica" naturale.

### 3.2 Limiti
- Più costoso ($O(K d^2)$ per iterazione) per via del calcolo della covarianza.
- Richiede `n / K > d` per stimare bene la covarianza.
- `covariance_type='full'` può overfit se $d$ è grande; usare `'diag'` o `'tied'` per regolarizzare.

### 3.3 Quando preferire GMM a KMeans

- I cluster hanno **forme ellissoidali** (le feature hanno scale diverse residue al post-scaling, o correlazioni forti).
- Si vuole un **anomaly score probabilistico** invece che una distanza puramente geometrica.
- Si vogliono **soft cluster** (per gestire ambiguità ai bordi).

In Ames-IoT, dopo `StandardScaler`, KMeans funziona bene perché i regimi sono ben separati. GMM aggiungerebbe ~0.02 di ROC-AUC al costo di 5× il tempo di training. Trade-off non favorevole per il PW didattico.

## 4. Scelta di K

Tre approcci:

### 4.1 Elbow method
Plotta inertia (somma dei quadrati delle distanze) vs K. La curva è monotona decrescente; si cerca il "gomito" — il punto di flesso dove l'aggiungere un cluster non riduce significativamente l'inertia.

```
inertia
  │ \
  │  \
  │   \___        ← elbow qui
  │       \___
  │           \___
  └────────────────── K
```

Limite: il gomito è soggettivo, spesso poco netto su dataset reali.

### 4.2 Silhouette score
Per ogni punto $i$:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

con $a(i)$ = distanza media dai punti dello stesso cluster, $b(i)$ = distanza media dal cluster più vicino diverso. Range $[-1, +1]$:
- $+1$: punto ben dentro il proprio cluster.
- $0$: sul confine.
- $-1$: assegnato al cluster sbagliato.

Si calcola la media su tutti i punti e si sceglie il K che la massimizza.

**Vantaggio**: oggettivo e numericamente comparabile.
**Svantaggio**: $O(n^2)$ → su 230k righe richiede ~1 minuto. Soluzione: campionare 5000 punti.

### 4.3 BIC / AIC (solo GMM)
$$
\text{BIC} = -2 \log L + p \log n
$$

con $L$ likelihood, $p$ numero parametri, $n$ punti. Penalizza modelli troppo complessi. Si sceglie il K con BIC minimo.

Per la nostra pipeline usiamo **silhouette** sui K candidati ${3, 4, 5, 6, 7, 8, 10}$. Il PW privilegia interpretabilità: con 3 regimi noti, K=3-5 è atteso.

## 5. Distanze per anomaly scoring

Una volta addestrato il modello, lo score di anomalia per un nuovo punto $\mathbf{x}$ è la distanza al centroide più vicino:

$$
\text{score}(\mathbf{x}) = \min_{k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2
$$

Per GMM, lo score è la **negativa log-likelihood**:

$$
\text{score}(\mathbf{x}) = -\log p(\mathbf{x} \mid \text{model})
$$

Più alto = più anomalo. La soglia separa "normale" da "anomalo".

## 6. Riferimenti

- **MacQueen, J.** (1967), *Some Methods for Classification and Analysis of Multivariate Observations*.
- **Sculley, D.** (2010), *Web-Scale K-Means Clustering*, WWW '10.
- **Bishop, C.** (2006), *Pattern Recognition and Machine Learning*, cap. 9 (mixture models, EM).
- **sklearn user guide**: [Clustering](https://scikit-learn.org/stable/modules/clustering.html).
