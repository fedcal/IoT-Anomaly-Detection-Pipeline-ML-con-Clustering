# Feature engineering per time-series IoT

## 1. Perché non basta lo "stato istantaneo"

Un singolo campione di sensori (rpm, temp, vibrazioni a un istante $t$) descrive lo **stato attuale**, ma non dice nulla sulla **dinamica** — come quei sensori stiano evolvendo.

Le anomalie industriali sono raramente eventi istantanei. Un cuscinetto che inizia a degradarsi mostra:

- Un lieve aumento del **livello medio** di vibrazione (rolling mean).
- Un aumento della **varianza locale** (rolling std).
- **Spike** sporadici sopra il rumore di fondo (z-score elevato).
- **Trend** (derivata prima positiva persistente).

Un classifier che vede solo lo stato istantaneo perde tutte queste informazioni. Aggiungere feature derivate temporali è quasi sempre il miglior single-shot improvement nel detection di anomalie su time-series.

## 2. Le quattro feature classiche

### 2.1 Rolling mean

$$
\bar{x}_t^{(w)} = \frac{1}{w} \sum_{i=t-w+1}^{t} x_i
$$

- **Significato**: "valore tipico" su una finestra di $w$ campioni.
- **Cattura**: trend lenti, shift di livello.
- **Larghezza tipica**: 5-30 minuti (a seconda della velocità del processo).

### 2.2 Rolling std

$$
\sigma_t^{(w)} = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^{t} (x_i - \bar{x}_t^{(w)})^2}
$$

- **Cattura**: variabilità locale. Una macchina sana ha rumore più o meno costante; un cuscinetto in degrado mostra varianza crescente.
- **Trade-off w**: piccolo → reattivo ma rumoroso; grande → stabile ma lento.

### 2.3 Differenza prima (derivata discreta)

$$
\Delta x_t = x_t - x_{t-1}
$$

- **Cattura**: cambi rapidi (transitori, accelerazioni, picchi).
- **Pro**: rileva eventi puntuali che sono invisibili alla rolling mean.
- **Contro**: amplifica il rumore. Spesso si combina con uno **smoothing** (es. mean della derivata su finestra).

### 2.4 Z-score locale

$$
z_t = \frac{x_t - \bar{x}_t^{(w)}}{\sigma_t^{(w)}}
$$

- **Significato**: quanto il valore istantaneo si discosta dalla "normalità locale", in unità di deviazioni standard.
- **Cattura**: deviazioni puntuali rispetto al regime corrente.
- **Vantaggio**: **scala-invariante** — robusto a cambi di regime di carico.

Esempio: un motore in idle ha vib_rms ~0.01 ± 0.005 (z-score 0). Se a regime di lavoro vib_rms = 0.05 ± 0.002, il valore 0.05 è ancora "normale" (z-score basso) — la stessa rolling mean lo segnerebbe come anomalo se vista dalla scala dell'idle.

## 3. Calcolo "per asset"

Critico: tutte le rolling vanno calcolate **separatamente per ciascun asset**. Mescolare le serie di asset diversi (con `df.rolling(15).mean()` puro) produce valori inutili dove la finestra straddla il confine fra due asset.

Implementazione corretta:

```python
df.groupby('asset_id', sort=False)[col]
  .rolling(window).mean()
  .reset_index(level=0, drop=True)
```

`sort=False` per non riordinare il DataFrame; `reset_index` per riallineare l'output al DataFrame originale.

## 4. No-leakage by construction

La nostra implementazione (`pandas.Series.rolling(w)`) è **back-looking**: al tempo $t$ aggrega l'intervallo $[t-w+1, t]$, mai $[t, t+w-1]$. Questo è essenziale per evitare leakage temporale.

**Errori comuni che generano leakage:**

- Usare `center=True`: la finestra include campioni futuri.
- Usare `df.expanding()` su tutto il DataFrame ordinato senza groupby: include campioni di asset successivi.
- Calcolare statistiche globali (`df.x.std()`) e usarle come feature: include il futuro.

La nostra pipeline è per costruzione safe perché:

1. `groupby('asset_id')` isola le serie.
2. `rolling(w)` è back-looking di default.
3. Le rolling sono dentro un `BaseEstimator/TransformerMixin` → quando applicate al test (`transform`), non hanno informazioni del training.

## 5. Riduzione dimensionalità (PCA opzionale)

Dopo il feature engineering passiamo da 19 a ~50 colonne. Per dataset più piccoli (~10k righe) la PCA è quasi obbligatoria; con 230k righe è opzionale.

**PCA in sintesi:**
- Trova $d_{out}$ direzioni ortogonali che catturano la massima varianza.
- Ogni direzione è una combinazione lineare delle feature originali.
- Riduce la dimensionalità mantenendo (per esempio) il 95% della varianza.

Quando aiuta:
- Molte feature collineari (lo scaling le rende uguali in scala ma non in informazione).
- Si vuole visualizzare i cluster in 2-3 dimensioni.
- Si vuole velocizzare l'inferenza.

Quando non serve:
- Le feature originali sono già informative e non collineari.
- Il numero di feature è gestibile per il modello.

Nella nostra pipeline è opzionale (`config.use_pca=True`).

## 6. Riferimenti

- **Kanawaday & Sane** (2017), *Machine Learning for Predictive Maintenance...*, ICUFN.
- **Hyndman & Athanasopoulos** (2021), *Forecasting: Principles and Practice*, cap. 6 (decomposizione di serie temporali).
- **Box & Jenkins** (1976), *Time Series Analysis*: classico delle finestre mobili.
