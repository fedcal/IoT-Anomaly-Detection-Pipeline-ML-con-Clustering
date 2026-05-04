"""Inferenza: detect_anomalies() su nuovi dati IoT.

Carica il modello completo (preprocessor + scaler + cluster + soglia)
serializzato e restituisce un DataFrame arricchito con score e flag.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH: Path = MODELS_DIR / "anomaly_detector.joblib"


@lru_cache(maxsize=4)
def _load_artifacts(model_path: str) -> dict:
    """Carica gli artefatti serializzati: feature_engineer, scaler, detector, modeling_features.

    Cache LRU per evitare I/O ripetuti su chiamate successive.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modello non trovato a {path}. "
            "Esegui prima la pipeline di training: `iot-detect`."
        )
    logger.info("Carico artefatti da %s", path)
    return joblib.load(path)


def detect_anomalies(
    df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    return_only_anomalies: bool = False,
) -> pd.DataFrame:
    """Predice anomalie per ogni riga del DataFrame.

    Args:
        df: DataFrame con almeno `timestamp`, `asset_id`, sensori.
            Le colonne ground-truth (anomaly_label, fault_code_true)
            sono opzionali — non usate per la predizione.
        model_path: path al joblib del detector addestrato.
        return_only_anomalies: se True, ritorna solo le righe predette
            anomale.

    Returns:
        DataFrame copia di `df` con due colonne aggiunte:
            - `anomaly_score`: distanza dal cluster (più alto = più anomalo).
            - `anomaly_pred`: 1 se anomalo, 0 altrimenti.
    """
    artifacts = _load_artifacts(str(model_path))
    feature_engineer = artifacts["feature_engineer"]
    scaler = artifacts["scaler"]
    detector = artifacts["detector"]
    modeling_features: list[str] = artifacts["modeling_features"]
    wrangling_columns: tuple[str, ...] = tuple(artifacts.get("wrangling_columns", ()))
    pca = artifacts.get("pca", None)

    # Pipeline di trasformazione coerente col training: wrangling +
    # feature engineering + scaling. Riprodurre lo stesso ordine
    # del training è essenziale per evitare drift di feature.
    from .wrangling import add_missing_flags, fill_missing_per_asset
    df_clean = add_missing_flags(df, columns=wrangling_columns) if wrangling_columns else df
    df_clean = fill_missing_per_asset(df_clean, columns=wrangling_columns) if wrangling_columns else df_clean
    df_fe = feature_engineer.transform(df_clean)
    X = df_fe[modeling_features].to_numpy()
    X_scaled = scaler.transform(X)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    scores = detector.score(X_scaled)
    preds = (scores >= detector.threshold).astype(int)

    out = df.copy()
    out["anomaly_score"] = scores
    out["anomaly_pred"] = preds

    if return_only_anomalies:
        return out.loc[out["anomaly_pred"] == 1].copy()
    return out


__all__ = ["detect_anomalies", "DEFAULT_MODEL_PATH"]
