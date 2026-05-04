"""Genera i 4 notebook didattici in `notebooks/` per PW2 IoT anomaly.

Lo script è la SORGENTE DI VERITÀ dei notebook: per modificarli si
edita questo file e si rilancia.
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write_nb(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.write(nb, NB_DIR / name)
    print(f"[OK] notebooks/{name}  ({len(cells)} celle)")


# 01 — EDA
nb01 = [
    md(
        "# 01 — Esplorazione del dataset IoT\n\n"
        "## Obiettivi didattici\n\n"
        "1. Comprendere la struttura del dataset (16 asset, 10 giorni, 1 minuto di sampling).\n"
        "2. Identificare i tre tipi di colonne: **sensoriali**, **di contesto**, **label**.\n"
        "3. Quantificare i valori mancanti e visualizzare la copertura temporale.\n"
        "4. Verificare la regolarità del campionamento (1 minuto = 1 campione, sempre).\n"
        "5. Visualizzare la prevalenza di anomalie e fault per asset/regime.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 110\n"
        "from iot_anomaly.data import load_raw\n"
        "from iot_anomaly.config import SENSOR_COLUMNS, CONTEXT_COLUMNS, LABEL_COLUMNS\n"
    ),
    code(
        "df = load_raw()\n"
        "print(f'Shape: {df.shape}')\n"
        "print(f'Periodo: {df.timestamp.min()} → {df.timestamp.max()}')\n"
        "print(f'Asset: {df.asset_id.nunique()}, regime: {df.regime.nunique()}')\n"
        "df.head(3)"
    ),
    md(
        "## Le tre famiglie di colonne\n\n"
        "**Sensoriali** (caratterizzano lo stato della macchina):  \n"
        f"`{', '.join(s for s in ['rpm', 'current_a', 'pressure_bar', 'flow_lpm', 'temp_c', 'vib_rms', 'vib_crest', 'vib_kurtosis'])}`\n\n"
        "**Di contesto** (descrivono regime e ambiente):  \n"
        "`regime, ambient_temp_c, humidity_pct, load_pct`\n\n"
        "**Label / ground-truth** (DA NON USARE per il training):  \n"
        "`fault_code_true, fault_type_true, anomaly_label`\n"
    ),
    code(
        "missing_pct = df.isna().mean().sort_values(ascending=False) * 100\n"
        "missing_pct = missing_pct[missing_pct > 0]\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.barh(missing_pct.index[::-1], missing_pct.values[::-1])\n"
        "ax.set_xlabel('% missing'); ax.set_title('Valori mancanti per colonna')\n"
        "for i, v in enumerate(missing_pct.values[::-1]):\n"
        "    ax.text(v, i, f' {v:.2f}%', va='center')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## Distribuzione temporale\n\n"
        "Aspettiamo: 14400 campioni per asset (= 10 giorni × 1440 minuti). "
        "Tutti gli asset coprono lo stesso periodo."
    ),
    code(
        "asset_counts = df.groupby('asset_id').size()\n"
        "print('campioni per asset:')\n"
        "print(asset_counts.describe())\n"
        "assert (asset_counts == 14400).all(), 'Asset con copertura disallineata!'\n"
        "print('[OK] tutti gli asset hanno 14400 campioni.')"
    ),
    md(
        "## Distribuzione di regimi e label\n\n"
        "I tre regimi (0, 1, 2) corrispondono a stati operativi distinti — "
        "questo è il segnale più forte che il clustering dovrebbe catturare. "
        "Le label `anomaly_label` (~2.4%) e `fault_code_true != 0` (~4%) sono "
        "rare e parziali: validazione \"debole\"."
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "df.regime.value_counts().sort_index().plot.bar(ax=axes[0], title='Regime', color='C0')\n"
        "df.anomaly_label.value_counts().plot.bar(ax=axes[1], title='anomaly_label', color='C3')\n"
        "(df.fault_code_true != 0).value_counts().plot.bar(ax=axes[2], title='fault_code != 0', color='C2')\n"
        "for a in axes: a.tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'Anomalie: {df.anomaly_label.mean():.2%} dei campioni')\n"
        "print(f'Fault: {(df.fault_code_true != 0).mean():.2%} dei campioni')"
    ),
    md(
        "## Visualizzare un asset nel tempo\n\n"
        "Per intuizione visiva: prendiamo l'asset 0 e plottiamo `vib_rms`, "
        "evidenziando i punti etichettati come anomalia."
    ),
    code(
        "asset_id = 0\n"
        "sub = df[df.asset_id == asset_id].sort_values('timestamp')\n"
        "anom_pts = sub[sub.anomaly_label == 1]\n"
        "fig, ax = plt.subplots(figsize=(13, 4))\n"
        "ax.plot(sub.timestamp, sub.vib_rms, alpha=0.6, lw=0.6, label='vib_rms')\n"
        "ax.scatter(anom_pts.timestamp, anom_pts.vib_rms, c='red', s=14,\n"
        "           label=f'anomaly_label=1 ({len(anom_pts)})', zorder=5)\n"
        "ax.set_xlabel('Tempo'); ax.set_ylabel('vib_rms')\n"
        "ax.set_title(f'Asset {asset_id}: vib_rms nel tempo + anomalie note')\n"
        "ax.legend(); plt.xticks(rotation=20)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Conclusioni dell'EDA\n\n"
        "- **3 regimi operativi distinti** → il clustering dovrebbe trovare almeno "
        "  3 cluster \"normali\" (uno per regime).\n"
        "- **Missing ~3.85%** concentrati sui sensori → forward-fill per asset.\n"
        "- **Label parziali**: validazione possibile ma debole. Non usarle per training.\n"
        "- **Sampling regolare (1 minuto)** → ok rolling window in unità di minuti.\n\n"
        "Vai al notebook **02_features_pipeline**.\n"
    ),
]
write_nb("01_eda.ipynb", nb01)


# 02 — Features e pipeline
nb02 = [
    md(
        "# 02 — Wrangling e feature engineering temporale\n\n"
        "## Obiettivi didattici\n\n"
        "1. Distinguere wrangling \"deterministico\" (ok prima dello split) da imputazione "
        "statistica (deve stare dentro la pipeline).\n"
        "2. Generare feature temporali (rolling, diff, z-score) per asset.\n"
        "3. Verificare che le feature derivate non leakino il futuro.\n"
        "4. Eseguire lo split temporale (no shuffle!) sui primi 7 giorni come training.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from iot_anomaly.data import load_raw, time_split\n"
        "from iot_anomaly.wrangling import add_missing_flags, fill_missing_per_asset\n"
        "from iot_anomaly.features import TimeSeriesFeatureEngineer, select_modeling_features\n"
        "from iot_anomaly.config import DEFAULT_CONFIG, SENSOR_COLUMNS\n"
    ),
    code(
        "df = load_raw()\n"
        "cols = SENSOR_COLUMNS + ('ambient_temp_c', 'humidity_pct', 'load_pct')\n"
        "df = add_missing_flags(df, columns=cols)\n"
        "df = fill_missing_per_asset(df, columns=cols)\n"
        "fe = TimeSeriesFeatureEngineer(window=15)\n"
        "df_fe = fe.fit_transform(df)\n"
        "print(f'Colonne: {df.shape[1]} → {df_fe.shape[1]}')\n"
        "print(f'Feature derivate: {df_fe.shape[1] - df.shape[1]}')\n"
    ),
    md(
        "## Verifica anti-leakage delle rolling\n\n"
        "Una rolling che usa un campione del **futuro** sarebbe leakage. La nostra "
        "implementazione usa `pandas.Series.rolling(window).mean()` che è "
        "back-looking: al tempo *t* aggrega `[t-w+1, t]`, mai `[t, t+w-1]`. "
        "Verifichiamolo manualmente."
    ),
    code(
        "asset0 = df_fe[df_fe.asset_id == 0].iloc[:25][['timestamp','vib_rms','vib_rms_roll_mean']]\n"
        "asset0.head(20)"
    ),
    md(
        "## Time-aware split\n\n"
        "Splittiamo il dataset cronologicamente: i primi 7 giorni → training "
        "(considerati \"normali\"), gli ultimi 3 giorni → test. Tutti i 16 asset "
        "sono presenti in entrambi i set; cambia solo la finestra temporale."
    ),
    code(
        "train, test, cutoff = time_split(df_fe, DEFAULT_CONFIG)\n"
        "print(f'Cutoff: {cutoff}')\n"
        "print(f'train: {len(train):,} righe, test: {len(test):,} righe')\n"
        "print(f'asset in train: {train.asset_id.nunique()}, in test: {test.asset_id.nunique()}')\n"
        "print(f'anomaly_label train: {train.anomaly_label.mean():.2%}, test: {test.anomaly_label.mean():.2%}')\n"
    ),
    md(
        "## Feature di modellazione finali\n\n"
        "Le feature usate dal clustering combinano:\n"
        "- Sensori grezzi\n"
        "- Rolling mean/std/diff/zscore\n"
        "- Variabili di contesto (load_pct, humidity, ambient_temp)\n"
    ),
    code(
        "modeling = select_modeling_features(df_fe)\n"
        "print(f'{len(modeling)} feature di modellazione:')\n"
        "for c in modeling:\n"
        "    print(f'  - {c}')\n"
    ),
]
write_nb("02_features_pipeline.ipynb", nb02)


# 03 — Clustering e soglia
nb03 = [
    md(
        "# 03 — Clustering, scelta di K, soglia di anomalia\n\n"
        "## Obiettivi didattici\n\n"
        "1. Confrontare KMeans, MiniBatchKMeans e GaussianMixture.\n"
        "2. Scegliere K via silhouette + interpretazione qualitativa dei cluster.\n"
        "3. Calcolare la distribuzione delle distanze sul training.\n"
        "4. Definire la soglia di anomalia come percentile (99° tipico).\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from iot_anomaly.data import load_raw, time_split\n"
        "from iot_anomaly.wrangling import add_missing_flags, fill_missing_per_asset\n"
        "from iot_anomaly.features import TimeSeriesFeatureEngineer, select_modeling_features\n"
        "from iot_anomaly.clustering import (fit_minibatch_kmeans, select_k_by_silhouette,\n"
        "                                    KMEANS_N_CLUSTERS_RANGE)\n"
        "from iot_anomaly.scoring import fit_anomaly_detector\n"
        "from iot_anomaly.config import DEFAULT_CONFIG, SENSOR_COLUMNS\n"
        "\n"
        "df = load_raw()\n"
        "cols = SENSOR_COLUMNS + ('ambient_temp_c', 'humidity_pct', 'load_pct')\n"
        "df = add_missing_flags(df, columns=cols)\n"
        "df = fill_missing_per_asset(df, columns=cols)\n"
        "df_fe = TimeSeriesFeatureEngineer(window=15).fit_transform(df)\n"
        "train, test, cutoff = time_split(df_fe, DEFAULT_CONFIG)\n"
        "feats = select_modeling_features(df_fe)\n"
        "scaler = StandardScaler().fit(train[feats])\n"
        "X_train = scaler.transform(train[feats])\n"
        "X_test = scaler.transform(test[feats])\n"
        "print(f'X_train.shape={X_train.shape}, X_test.shape={X_test.shape}')\n"
    ),
    md(
        "## Silhouette per la scelta di K\n\n"
        "Testiamo K ∈ {3, 4, 5, 6, 7, 8, 10}. Il silhouette score (range −1..+1) "
        "misura quanto i punti sono ben raggruppati: vicini al loro centroide e "
        "lontani dagli altri."
    ),
    code(
        "best_k, results = select_k_by_silhouette(X_train, k_range=KMEANS_N_CLUSTERS_RANGE)\n"
        "summary = pd.DataFrame([{'K': r.n_clusters, 'silhouette': r.silhouette,\n"
        "                          'inertia': r.inertia} for r in results])\n"
        "summary"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(summary.K, summary.silhouette, 'o-'); axes[0].set(title='Silhouette vs K', xlabel='K', ylabel='silhouette')\n"
        "axes[1].plot(summary.K, summary.inertia, 'o-'); axes[1].set(title='Inertia vs K (elbow)', xlabel='K', ylabel='inertia')\n"
        "for a in axes: a.grid(True, alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## Fit del modello finale e soglia\n\n"
        "Con il K scelto, addestriamo MiniBatchKMeans, calcoliamo le distanze "
        "sul training e fissiamo la soglia al 99° percentile."
    ),
    code(
        "result = fit_minibatch_kmeans(X_train, n_clusters=best_k)\n"
        "detector = fit_anomaly_detector(result.model, X_train, threshold_percentile=99.0)\n"
        "print(f'Modello: {result.model_name}, soglia={detector.threshold:.3f}')\n"
    ),
    code(
        "train_scores = detector.score(X_train)\n"
        "test_scores = detector.score(X_test)\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "ax.hist(train_scores, bins=80, alpha=0.6, density=True, label='train')\n"
        "ax.hist(test_scores, bins=80, alpha=0.6, density=True, label='test')\n"
        "ax.axvline(detector.threshold, color='red', ls='--', lw=1.5, label=f'soglia={detector.threshold:.2f}')\n"
        "ax.set(xlabel='distanza al cluster più vicino', ylabel='densità',\n"
        "       title='Distribuzione score di anomalia')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## Interpretazione cluster\n\n"
        "Aggreghiamo per cluster + regime per verificare che il clustering catturi "
        "i regimi operativi."
    ),
    code(
        "train_with_cluster = train.assign(cluster=result.labels)\n"
        "ct = pd.crosstab(train_with_cluster.cluster, train_with_cluster.regime)\n"
        "ct_norm = ct.div(ct.sum(axis=1), axis=0)\n"
        "print('% di ogni cluster nei tre regimi:')\n"
        "print((ct_norm * 100).round(1))\n"
    ),
]
write_nb("03_clustering_threshold.ipynb", nb03)


# 04 — Validazione
nb04 = [
    md(
        "# 04 — Validazione e inferenza\n\n"
        "## Obiettivi didattici\n\n"
        "1. Confrontare anomaly_pred con `anomaly_label` (ground-truth parziale).\n"
        "2. Calcolare metriche di anomaly detection (precision/recall, ROC-AUC, PR-AUC).\n"
        "3. Discutere il trade-off tra precision e recall in funzione della soglia.\n"
        "4. Visualizzare la mappa temporale delle anomalie predette.\n"
        "5. Usare la funzione `detect_anomalies()` per inferenza.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from iot_anomaly.pipeline import run_full_pipeline\n"
        "from iot_anomaly.evaluation import (evaluate, plot_score_distribution,\n"
        "                                     plot_pr_curve, plot_temporal_anomalies)\n"
        "from iot_anomaly.config import DEFAULT_CONFIG\n"
        "from iot_anomaly.inference import detect_anomalies\n"
    ),
    md(
        "## Esecuzione pipeline completa\n\n"
        "Usiamo `run_full_pipeline` con la config di default. Tempo atteso ~30s."
    ),
    code(
        "result = run_full_pipeline(config=DEFAULT_CONFIG, quick=True)\n"
        "print(f'Best K: {result[\"best_k\"]}')\n"
        "print(f'Soglia: {result[\"detector\"].threshold:.3f}')\n"
    ),
    md(
        "## Metriche di validazione\n\n"
        "Le label sono parziali: la vera prevalenza di anomalie potrebbe essere "
        "diversa. Le metriche vanno lette come **indicative**, non assolute."
    ),
    code(
        "rows = [{'set': 'train', **result['eval_train'].as_dict()},\n"
        "        {'set': 'test',  **result['eval_test'].as_dict()}]\n"
        "metrics_df = pd.DataFrame(rows).set_index('set')\n"
        "metrics_df.style.format({\n"
        "    'precision': '{:.3f}', 'recall': '{:.3f}', 'f1': '{:.3f}',\n"
        "    'roc_auc': '{:.3f}', 'pr_auc': '{:.3f}', 'fault_code_recall': '{:.3f}',\n"
        "})"
    ),
    md(
        "## Trade-off soglia: precision vs recall\n\n"
        "Variando il percentile della soglia possiamo regolare il bilancio "
        "fra falsi positivi e falsi negativi. Plot della curva PR per il test set."
    ),
    code(
        "test = result['test_df']\n"
        "y_true = test.anomaly_label.values\n"
        "y_score = test.anomaly_score.values\n"
        "fig = plot_pr_curve(y_true, y_score)\n"
        "plt.show()\n"
    ),
    md(
        "## Mappa temporale delle anomalie\n\n"
        "Heatmap: ogni riga è un asset, x = tempo, rosso = anomalia predetta. "
        "Permette di identificare *quando* e *quale asset* è andato fuori regime."
    ),
    code(
        "fig = plot_temporal_anomalies(test)\n"
        "plt.show()\n"
    ),
    md(
        "## Inferenza su nuovi dati\n\n"
        "`detect_anomalies(df)` accetta un DataFrame con le stesse colonne del "
        "training. Riapplica wrangling, feature engineering, scaling, scoring."
    ),
    code(
        "from iot_anomaly.data import load_raw, time_split\n"
        "df_full = load_raw()\n"
        "_, test_raw, _ = time_split(df_full, DEFAULT_CONFIG)\n"
        "sample = test_raw.sample(2000, random_state=42).reset_index(drop=True)\n"
        "out = detect_anomalies(sample)\n"
        "print(f'Anomalie predette: {int(out.anomaly_pred.sum())} / {len(out)}')\n"
        "print(f'Score range: {out.anomaly_score.min():.2f} → {out.anomaly_score.max():.2f}')\n"
        "out[['timestamp','asset_id','vib_rms','anomaly_score','anomaly_pred','anomaly_label']].head(10)"
    ),
    md(
        "## Conclusioni\n\n"
        "Su questo dataset, un MiniBatchKMeans con `K=5` e soglia al 99° percentile "
        "raggiunge **ROC-AUC ~0.88** e **F1 ~0.34** sul test set: discreto, considerando:\n\n"
        "- Le label sono **parziali** (non tutte le anomalie reali sono etichettate).\n"
        "- Il clustering **non vede** le label durante il training.\n"
        "- La prevalenza è **2.4%**: PR-AUC > 0.30 è 13× sopra il random.\n\n"
        "**Estensioni naturali**: GMM con BIC, DBSCAN, autoencoder per anomaly score "
        "non basato su distanza euclidea, soglia adattiva per asset.\n"
    ),
]
write_nb("04_validation_inference.ipynb", nb04)

print("\nTutti i 4 notebook generati in", NB_DIR)
