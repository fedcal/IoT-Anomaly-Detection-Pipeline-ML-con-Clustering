"""Anomaly scoring: distanza dal cluster più vicino + soglia su percentile.

L'idea: un punto è "normale" se sta vicino a uno dei centroidi appresi
sul training (= comportamento normale assunto). Più si allontana, più
è anomalo. La soglia separa "lontano normale" da "anomalo" sulla base
del percentile della distribuzione di distanze osservate sul training.

Distanze gestite:
- **Euclidea dal centroide più vicino**: standard per KMeans.
- **Negativa log-likelihood**: per GMM (il modello stesso fornisce
  un'analoga di "distanza" probabilistica).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetector:
    """Wrapper non-sklearn che porta soglia + scoring insieme al modello."""
    model: object
    threshold: float
    threshold_percentile: float
    train_score_distribution: np.ndarray  # per visualizzazione

    def score(self, X: np.ndarray) -> np.ndarray:
        """Distanza/score per ogni punto. Più alto = più anomalo."""
        return _score(self.model, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomalo, 0 = normale."""
        return (self.score(X) >= self.threshold).astype(int)


def _score(model: object, X: np.ndarray) -> np.ndarray:
    """Calcola lo score adatto al tipo di modello."""
    if isinstance(model, (KMeans, MiniBatchKMeans)):
        # Distanza al centroide più vicino. `transform` ritorna la
        # distanza a TUTTI i centroidi; min sull'asse 1 dà quella al
        # più vicino.
        return model.transform(X).min(axis=1)
    if isinstance(model, GaussianMixture):
        # Negative log-density: più alto = più "lontano" dalla densità.
        return -model.score_samples(X)
    raise TypeError(f"Modello non supportato per scoring: {type(model).__name__}")


def fit_anomaly_detector(
    model: object,
    X_train: np.ndarray,
    threshold_percentile: float = 99.0,
) -> AnomalyDetector:
    """Calcola la soglia come percentile delle distanze su train.

    Punto critico: il training è ASSUNTO normale (primi N giorni).
    Calcoliamo il percentile sulle distanze: il (100-p)% delle
    osservazioni di training sarà classificato anomalo. Default 99°
    percentile → solo l'1% dei punti di train (i più lontani dai
    propri centroidi) viene considerato anomalo. Sul test set la
    frazione effettiva può essere maggiore (se ci sono regimi
    veramente anomali).
    """
    scores = _score(model, X_train)
    threshold = float(np.percentile(scores, threshold_percentile))
    logger.info(
        "Soglia anomalia = percentile %.1f (= %.4f). "
        "Distribuzione train: min=%.4f, p50=%.4f, p90=%.4f, max=%.4f.",
        threshold_percentile, threshold,
        float(scores.min()), float(np.percentile(scores, 50)),
        float(np.percentile(scores, 90)), float(scores.max()),
    )
    return AnomalyDetector(
        model=model,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        train_score_distribution=scores,
    )


__all__ = ["AnomalyDetector", "fit_anomaly_detector"]
