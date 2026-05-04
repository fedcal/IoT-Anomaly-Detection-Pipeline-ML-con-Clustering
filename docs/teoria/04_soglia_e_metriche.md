---
layout: default
title: Soglia & metriche
parent: Teoria
nav_order: 4
math: mathjax
description: >-
  Scelta della soglia ottimale di anomaly detection: percentile vs
  business-driven; metriche per classi sbilanciate (precision, recall, F1,
  ROC-AUC, PR-AUC) e validazione vs ground-truth parziale.
---

# Soglia di anomalia, metriche e validazione
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Dalla "distanza" al "flag binario"

Il modello produce una **distanza** (o, equivalentemente, uno **score** continuo): più alto = più anomalo. Per usarla in produzione serve binarizzarla:

$$
\hat{y}(\mathbf{x}) = \begin{cases} 1 \text{ (anomalia)} & \text{se score}(\mathbf{x}) \geq \theta \\ 0 \text{ (normale)} & \text{altrimenti} \end{cases}
$$

La scelta della **soglia** $\theta$ è l'iperparametro più impattante dopo K. Determina il trade-off tra:

- **Falsi positivi (FP)**: campioni normali segnalati come anomali → spreco di interventi di manutenzione.
- **Falsi negativi (FN)**: anomalie reali mancate → guasti non rilevati, downtime.

## 2. Scelta della soglia

### 2.1 Approccio percentile (quello che usiamo)

Calcola la distribuzione delle distanze sul **training set** (assunto normale). La soglia è il **p-esimo percentile**:

$$
\theta = \text{percentile}_p(\{\text{score}(\mathbf{x}_i) : \mathbf{x}_i \in \text{train}\})
$$

Tipicamente $p = 99$ → solo l'1% dei punti di training è classificato come anomalo. Sul test, la frazione vera dipenderà dalla presenza di reali anomalie.

**Pro**:
- Indipendente dalle label (niente leakage del ground-truth).
- Auto-calibra la prevalenza attesa.
- Interpretabile: "voglio segnalare al massimo l'X% dei punti normali".

**Contro**:
- Assume che il training rappresenti il "normale". Se il training contiene anomalie, la soglia è sovrastimata.

### 2.2 Approccio basato su label (semi-supervised)

Se hai un sottoinsieme di label nel training:

- Calcola **F1 score** per ogni soglia candidata.
- Scegli la soglia che massimizza F1 sul sottoinsieme labelled.

Solo per validation **a posteriori**. Non possiamo usarlo in unsupervised puro come da PW.

### 2.3 Approccio business-driven

Stima il **costo** di FP e FN:

$$
\text{costo}_\theta = c_{FP} \cdot |\{ \hat{y}=1 \wedge y=0 \}| + c_{FN} \cdot |\{ \hat{y}=0 \wedge y=1 \}|
$$

Scegli la soglia che minimizza il costo atteso.

Esempio numerico: se un FP costa 100€ (intervento inutile) e un FN costa 10,000€ (downtime), la soglia ottima è bassa (preferiamo molti FP a uno solo FN).

## 3. Le metriche di valutazione

### 3.1 Confusion matrix

|              | Predetto 0 | Predetto 1 |
|--------------|------------|------------|
| **Vero 0**   | TN         | FP         |
| **Vero 1**   | FN         | TP         |

### 3.2 Precision, Recall, F1

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \frac{P \cdot R}{P + R}
$$

- **Precision**: fra i punti che ho segnalato anomali, quanti lo sono davvero.
- **Recall**: delle anomalie reali, quante ne ho catturate.
- **F1**: media armonica.

### 3.3 ROC-AUC

Curva: TPR (= recall) vs FPR al variare della soglia. AUC = area sotto la curva.

- $1.0$: classifier perfetto.
- $0.5$: random.
- $< 0.5$: peggio del random.

**Vantaggio**: indipendente dalla soglia. Misura la capacità del modello di separare le due classi.

**Svantaggio su classi sbilanciate**: con prevalenza 2.4%, anche un modello mediocre ottiene AUC alto perché i veri negativi sono tantissimi e dominano la curva. Va letto insieme alla PR-AUC.

### 3.4 PR-AUC (Precision-Recall AUC)

Curva: precision vs recall al variare della soglia. AUC = area sotto.

- **Più informativa di ROC-AUC su classi sbilanciate**.
- Random baseline = prevalenza della classe positiva (qui ~0.024).
- Ottenere PR-AUC=0.30 con prevalenza 0.024 significa essere **13× sopra random**.

## 4. Validazione su label parziali

Le label `anomaly_label` del PW sono dichiarate come "non necessariamente complete". Conseguenza: quando il modello segnala un punto anomalo che NON ha `anomaly_label=1`, non sappiamo se è:

- Un **vero falso positivo** (il modello ha sbagliato).
- Una **vera anomalia non etichettata** (il modello ha trovato qualcosa che gli annotatori hanno mancato).

Quindi:

- **La precision misurata è un lower bound** della precision vera.
- **La recall è inflated** se ci sono anomalie non etichettate (perché il denominatore include solo quelle conosciute).
- **`fault_code_recall`** (frazione di punti con `fault_code_true != 0` catturati) è più affidabile come stima della recall, perché `fault_code_true` è generato sinteticamente e copre tutte le anomalie funzionali.

## 5. Discussione critica dei risultati

In ML applicato non basta riportare le metriche: bisogna **interpretarle**.

Esempio sul nostro dataset (run quick):

- ROC-AUC test = 0.879 → forte capacità discriminativa.
- PR-AUC = 0.315 → 13× sopra random, ma non eccellente.
- Precision = 0.44, Recall = 0.28, F1 = 0.34 → soglia 99° forse troppo alta.

Interpretazione: il modello ha buon segnale, ma stiamo perdendo recall perché soglia conservativa. Abbassando al 95° percentile guadagniamo recall a costo di precision. La scelta "giusta" dipende dal costo business di FP vs FN.

## 6. Riferimenti

- **Davis & Goadrich** (2006), *The Relationship Between Precision-Recall and ROC Curves*, ICML.
- **Saito & Rehmsmeier** (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLoS ONE.
- **Powers, D.** (2011), *Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation*.
