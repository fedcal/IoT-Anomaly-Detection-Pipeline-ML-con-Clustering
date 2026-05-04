"""Configurazione globale della pipeline IoT anomaly detection.

Centralizza paths, costanti, iperparametri di default. Mantiene il
codice pulito (no magic numbers/path sparsi) e facilita esecuzioni
riproducibili.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

DATASET_FILENAME: str = "iot_synth_anomaly_clustering.csv"

# Colonne del dataset
TIMESTAMP_COL: str = "timestamp"
ASSET_COL: str = "asset_id"
REGIME_COL: str = "regime"

# Sensori "di macchina" — variabili che descrivono lo stato operativo.
SENSOR_COLUMNS: tuple[str, ...] = (
    "rpm",
    "current_a",
    "pressure_bar",
    "flow_lpm",
    "temp_c",
    "vib_rms",
    "vib_crest",
    "vib_kurtosis",
)

# Variabili di contesto — informazioni ambientali / di carico.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "ambient_temp_c",
    "humidity_pct",
    "load_pct",
    "regime",
)

# Colonne con label/ground-truth — usate SOLO per validazione.
# Regola fondamentale del PW: NON entrano nel training del clustering.
LABEL_COLUMNS: tuple[str, ...] = (
    "fault_code_true",
    "fault_type_true",
    "anomaly_label",
)

ID_COLUMNS: tuple[str, ...] = ("site_id", "line_id")

# Tutte le feature numeriche disponibili per la modellazione.
ALL_NUMERIC_FEATURES: tuple[str, ...] = SENSOR_COLUMNS + (
    "ambient_temp_c", "humidity_pct", "load_pct",
)

RANDOM_STATE: int = 42

# Split temporale di default: i primi 7 giorni come "comportamento
# normale" (training), gli ultimi 3 come test.
TRAIN_DAYS: int = 7

# Iperparametri del clustering K-Means.
KMEANS_N_CLUSTERS_RANGE: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 10)
KMEANS_N_CLUSTERS_DEFAULT: int = 5  # fallback

# Soglia anomalia: percentile della distribuzione di distanze sul training.
ANOMALY_THRESHOLD_PERCENTILE: float = 99.0

# Feature engineering rolling: dimensione finestra in minuti.
ROLLING_WINDOW_MIN: int = 15


@dataclass(frozen=True)
class PipelineConfig:
    """Iperparametri e flag della pipeline (immutabile)."""
    random_state: int = RANDOM_STATE
    train_days: int = TRAIN_DAYS
    rolling_window_min: int = ROLLING_WINDOW_MIN
    kmeans_n_clusters: int | None = None  # None → autoscelta da silhouette
    anomaly_threshold_percentile: float = ANOMALY_THRESHOLD_PERCENTILE
    use_pca: bool = False
    pca_n_components: int = 8
    n_jobs: int = -1
    verbose: int = 1


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
    "REPORTS_DIR", "FIGURES_DIR", "MODELS_DIR",
    "DATASET_FILENAME",
    "TIMESTAMP_COL", "ASSET_COL", "REGIME_COL",
    "SENSOR_COLUMNS", "CONTEXT_COLUMNS", "LABEL_COLUMNS", "ID_COLUMNS",
    "ALL_NUMERIC_FEATURES",
    "RANDOM_STATE", "TRAIN_DAYS",
    "KMEANS_N_CLUSTERS_RANGE", "KMEANS_N_CLUSTERS_DEFAULT",
    "ANOMALY_THRESHOLD_PERCENTILE", "ROLLING_WINDOW_MIN",
    "PipelineConfig", "DEFAULT_CONFIG",
]
