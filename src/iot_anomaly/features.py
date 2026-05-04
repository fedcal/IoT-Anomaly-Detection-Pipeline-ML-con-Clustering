"""Feature engineering temporale per time-series IoT.

Le feature derivate catturano la **dinamica** dei segnali, non solo
i valori istantanei. Questo è essenziale perché:

- Un valore "alto" non è anomalo di per sé (un motore al massimo carico
  ha rpm/temp alti); è anomalo *l'andamento*.
- Le anomalie raramente sono spike singoli — sono trend o cambi di
  varianza che si manifestano su finestre di alcuni minuti.

Le feature aggiunte (con default `window=15` minuti):

- **Rolling mean / std**: livello e variabilità locale.
- **Diff prima**: derivata discreta — cattura cambi rapidi.
- **Z-score locale**: `(x − rolling_mean) / rolling_std` — normalizza
  rispetto al regime corrente, evidenziando deviazioni puntuali.

Tutte le rolling sono calcolate **per asset_id**: i 16 asset hanno
storie indipendenti.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ASSET_COL, SENSOR_COLUMNS

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Aggiunge feature temporali rolling e di derivata al DataFrame.

    L'implementazione è in pandas (non in sklearn standard) perché
    serve la `groupby(asset_id)` per non mescolare le serie. Per
    questo NON va usata dentro un `ColumnTransformer` standard:
    va invocata sul DataFrame *prima* dello scaling.

    Args:
        sensor_columns: colonne sensoriali su cui calcolare le rolling.
        window: dimensione finestra in numero di campioni (1 campione = 1 minuto).
        add_diff: aggiunge la differenza prima `x_t - x_{t-1}`.
        add_zscore: aggiunge il z-score locale `(x - mean) / std`.
    """

    def __init__(
        self,
        sensor_columns: tuple[str, ...] = SENSOR_COLUMNS,
        window: int = 15,
        add_diff: bool = True,
        add_zscore: bool = True,
    ) -> None:
        self.sensor_columns = sensor_columns
        self.window = window
        self.add_diff = add_diff
        self.add_zscore = add_zscore

    def fit(self, X: pd.DataFrame, y=None) -> "TimeSeriesFeatureEngineer":
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TimeSeriesFeatureEngineer richiede un pd.DataFrame.")
        if ASSET_COL not in X.columns:
            raise KeyError(f"Colonna `{ASSET_COL}` mancante (necessaria per groupby).")

        out = X.copy()
        cols = [c for c in self.sensor_columns if c in out.columns]

        # Rolling mean e std per asset.
        # `min_periods=1` per non perdere righe iniziali; le prime
        # `window-1` righe avranno statistiche calcolate su meno
        # campioni, ma sono comunque utilizzabili.
        groups = out.groupby(ASSET_COL, sort=False, group_keys=False)
        for c in cols:
            roll = groups[c].rolling(self.window, min_periods=1)
            out[f"{c}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
            out[f"{c}_roll_std"] = roll.std(ddof=0).fillna(0.0).reset_index(level=0, drop=True)

        if self.add_diff:
            for c in cols:
                out[f"{c}_diff"] = (
                    groups[c].diff().fillna(0.0)
                )

        if self.add_zscore:
            for c in cols:
                mean = out[f"{c}_roll_mean"]
                std = out[f"{c}_roll_std"].replace(0.0, np.nan)
                out[f"{c}_zscore"] = ((out[c] - mean) / std).fillna(0.0)

        added = [c for c in out.columns if c not in X.columns]
        logger.info("TimeSeriesFeatureEngineer: aggiunte %d feature derivate.", len(added))
        return out


def select_modeling_features(
    df: pd.DataFrame,
    sensor_columns: tuple[str, ...] = SENSOR_COLUMNS,
    include_context: bool = True,
) -> list[str]:
    """Seleziona le colonne numeriche utili per il clustering.

    Esclude esplicitamente: `timestamp`, `asset_id`, label.
    Include: sensori grezzi + tutte le feature derivate (suffix
    `_roll_mean`, `_roll_std`, `_diff`, `_zscore`).
    """
    base = list(sensor_columns)
    derived = [
        c for c in df.columns
        if any(c.endswith(suffix) for suffix in ("_roll_mean", "_roll_std", "_diff", "_zscore"))
    ]
    context = ["ambient_temp_c", "humidity_pct", "load_pct"] if include_context else []

    selected = [c for c in (base + derived + context) if c in df.columns]
    return selected


__all__ = ["TimeSeriesFeatureEngineer", "select_modeling_features"]
