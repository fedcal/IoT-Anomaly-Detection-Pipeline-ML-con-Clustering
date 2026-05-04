"""Pipeline ML per anomaly detection su time-series IoT industriali.

API pubbliche principali:

    from iot_anomaly.pipeline import run_full_pipeline
    from iot_anomaly.inference import detect_anomalies
    from iot_anomaly.data import load_raw

Per il dettaglio architetturale e teorico, vedi `docs/`.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
