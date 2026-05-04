"""Wrangling: gestione missing values e outlier preliminari.

Per il dataset IoT, ~3.85% dei valori sono mancanti, concentrati sui
sensori. Strategia:

1. **Forward-fill per asset**: serie temporali a 1 minuto → un valore
   mancante isolato è probabilmente un buco di trasmissione, non un
   guasto. Forward-fill (per asset, non globale) lo riempie con
   l'ultimo valore valido. Limit configurabile.
2. **Backward-fill come fallback** per i primi N campioni di un asset
   (forward-fill non li raggiunge).
3. **Rimanenti NaN → 0** dopo wrangling (sentinella esplicita).

Outlier: NON rimuoviamo outlier in fase di wrangling — gli outlier
SONO il segnale che vogliamo rilevare. Rimuoverli sarebbe data leakage
sul target.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import ASSET_COL, SENSOR_COLUMNS, TIMESTAMP_COL

logger = logging.getLogger(__name__)


def fill_missing_per_asset(
    df: pd.DataFrame,
    columns: tuple[str, ...] = SENSOR_COLUMNS,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Forward-fill (e poi back-fill) per asset, sui buchi sensoriali.

    Args:
        df: DataFrame ordinato per (asset_id, timestamp).
        columns: colonne su cui applicare il fill.
        ffill_limit: numero massimo di righe consecutive da riempire
            in avanti. Oltre, lascia NaN (sospetto guasto del sensore).

    Returns:
        Nuovo DataFrame (immutabilità).
    """
    out = df.copy()
    cols_to_fill = [c for c in columns if c in out.columns]

    out[cols_to_fill] = (
        out
        .groupby(ASSET_COL, sort=False)[cols_to_fill]
        .transform(lambda s: s.ffill(limit=ffill_limit).bfill(limit=ffill_limit))
    )

    remaining = out[cols_to_fill].isna().sum().sum()
    logger.info(
        "fill_missing_per_asset: ffill/bfill (limit=%d). NaN residui: %d.",
        ffill_limit, int(remaining),
    )
    # I residui sono buchi lunghi: probabile sensore offline. Mettiamo 0
    # come sentinella (più 1-hot flag in features.py).
    out[cols_to_fill] = out[cols_to_fill].fillna(0.0)
    return out


def add_missing_flags(
    df: pd.DataFrame,
    columns: tuple[str, ...] = SENSOR_COLUMNS,
) -> pd.DataFrame:
    """Per ogni colonna sensoriale, aggiunge una flag `<col>_was_nan`.

    Permette al modello di sapere quando un valore è stato sintetizzato
    (era NaN originale) — informazione utile per evitare di trattare
    sensori spenti come se fossero attivi a 0.

    NOTA: va chiamata PRIMA di `fill_missing_per_asset`, altrimenti
    i NaN sono già stati riempiti.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_was_nan"] = out[col].isna().astype(int)
    return out


__all__ = ["fill_missing_per_asset", "add_missing_flags"]
