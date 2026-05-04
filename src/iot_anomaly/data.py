"""Caricamento e split temporale del dataset IoT.

Tutto ciò che riguarda I/O dati grezzi e la separazione train/test
**rispettando l'ordine temporale** vive qui.

Punto critico didattico: in regimi di anomaly detection NON si usa
mai uno split casuale — si usa uno split TEMPORALE che assume la
finestra iniziale come rappresentativa del comportamento "normale".
Sviluppare un modello su dati "del futuro" e testarlo "nel passato"
è la forma più subdola di leakage in ML su time-series.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from .config import (
    ASSET_COL,
    DATASET_FILENAME,
    RAW_DIR,
    TIMESTAMP_COL,
    PipelineConfig,
    DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)

EXPECTED_ROWS: Final[int] = 230_400
EXPECTED_COLS: Final[int] = 19
EXPECTED_ASSETS: Final[int] = 16


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Carica il CSV grezzo, parse timestamp, ordina per (asset, time).

    Mantiene tutte le colonne incluse le label (servono per la
    validazione, NON per il training del clustering).
    """
    if path is None:
        path = RAW_DIR / DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato a {path}. "
            f"Atteso il file `{DATASET_FILENAME}` (33 MB) fornito col PW."
        )
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    if df.shape[0] != EXPECTED_ROWS:
        raise ValueError(f"Numero righe inatteso: {df.shape[0]} (attese {EXPECTED_ROWS})")
    if df.shape[1] != EXPECTED_COLS:
        raise ValueError(f"Numero colonne inatteso: {df.shape[1]} (attese {EXPECTED_COLS})")

    # Ordina cronologicamente per asset (le statistiche rolling
    # dipendono dall'ordine; senza sort si ottengono valori mescolati).
    df = df.sort_values([ASSET_COL, TIMESTAMP_COL]).reset_index(drop=True)

    n_assets = df[ASSET_COL].nunique()
    if n_assets != EXPECTED_ASSETS:
        raise ValueError(f"Numero asset inatteso: {n_assets} (attesi {EXPECTED_ASSETS})")

    logger.info(
        "Dataset caricato: %d righe × %d colonne, %d asset, periodo %s → %s.",
        df.shape[0], df.shape[1], n_assets,
        df[TIMESTAMP_COL].min().date(), df[TIMESTAMP_COL].max().date(),
    )
    return df


def time_split(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split TEMPORALE: i primi `train_days` giorni → training (presunto normale).

    Returns:
        (df_train, df_test, cutoff_timestamp)

    Lo split è fatto sul `timestamp` GLOBALE: il cutoff è il giorno
    `min_date + train_days`. Tutti gli asset condividono lo stesso
    cutoff, così il training contiene `train_days × 16 asset × 1440
    minuti` punti, e tutti gli asset sono presenti sia in train che
    in test.
    """
    if TIMESTAMP_COL not in df.columns:
        raise KeyError(f"Colonna `{TIMESTAMP_COL}` mancante.")
    min_ts = df[TIMESTAMP_COL].min()
    cutoff = min_ts + pd.Timedelta(days=config.train_days)

    train = df.loc[df[TIMESTAMP_COL] < cutoff].copy()
    test = df.loc[df[TIMESTAMP_COL] >= cutoff].copy()

    logger.info(
        "Time-split: cutoff=%s. Train=%d (%.1f%%), Test=%d (%.1f%%).",
        cutoff,
        len(train), len(train) / len(df) * 100,
        len(test), len(test) / len(df) * 100,
    )
    return train, test, cutoff


__all__ = ["load_raw", "time_split", "EXPECTED_ROWS", "EXPECTED_COLS", "EXPECTED_ASSETS"]
