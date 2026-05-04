"""Modelli di clustering per la pipeline.

Tre opzioni didatticamente complementari:

- **MiniBatchKMeans**: per dataset grandi (>200k righe) è 10-100×
  più veloce di KMeans standard. Risultati equivalenti in pratica.
- **KMeans** (full batch): più stabile ma più lento, usato come
  alternativa per validare il MiniBatch.
- **GaussianMixture (GMM)**: clusters ellissoidali con covarianza
  full → cattura cluster non sferici (es. quando le feature hanno
  scale o correlazioni diverse residue al post-scaling).

Selezione di K (numero di cluster) automatica: testiamo K nel range
configurato e scegliamo quello con miglior **silhouette score**.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .config import KMEANS_N_CLUSTERS_RANGE, RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Output dell'addestramento del clustering."""
    model_name: str
    model: object
    n_clusters: int
    silhouette: float
    inertia: float | None
    labels: np.ndarray


def fit_minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> ClusteringResult:
    """Addestra MiniBatchKMeans con `n_clusters` centroidi.

    `batch_size=4096` è un valore tipico per dataset di queste
    dimensioni: abbastanza grande da avere statistiche stabili,
    abbastanza piccolo da mantenere la velocità del minibatch.
    """
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        batch_size=4096,
        max_iter=300,
    )
    labels = model.fit_predict(X)
    # Silhouette su tutto il dataset è O(n²) — sample 5000 punti.
    sil = _silhouette_sample(X, labels, sample_size=5000, random_state=random_state)
    return ClusteringResult(
        model_name=f"MiniBatchKMeans(K={n_clusters})",
        model=model,
        n_clusters=n_clusters,
        silhouette=sil,
        inertia=float(model.inertia_),
        labels=labels,
    )


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> ClusteringResult:
    """KMeans full-batch."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=500,
    )
    labels = model.fit_predict(X)
    sil = _silhouette_sample(X, labels, sample_size=5000, random_state=random_state)
    return ClusteringResult(
        model_name=f"KMeans(K={n_clusters})",
        model=model,
        n_clusters=n_clusters,
        silhouette=sil,
        inertia=float(model.inertia_),
        labels=labels,
    )


def fit_gmm(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> ClusteringResult:
    """Gaussian Mixture Model con `n_clusters` componenti.

    Utile quando i cluster hanno forme ellissoidali. Trade-off:
    O(n × K × d²) — più lento di KMeans su molte feature.
    """
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        max_iter=200,
        n_init=3,
    )
    labels = model.fit_predict(X)
    sil = _silhouette_sample(X, labels, sample_size=5000, random_state=random_state)
    return ClusteringResult(
        model_name=f"GMM(K={n_clusters})",
        model=model,
        n_clusters=n_clusters,
        silhouette=sil,
        inertia=None,
        labels=labels,
    )


def select_k_by_silhouette(
    X: np.ndarray,
    k_range: tuple[int, ...] = KMEANS_N_CLUSTERS_RANGE,
    fit_fn=fit_minibatch_kmeans,
    random_state: int = RANDOM_STATE,
) -> tuple[int, list[ClusteringResult]]:
    """Sceglie K maximizzando la silhouette su un sample.

    Returns:
        (best_k, all_results_ordered_by_k).

    *Trade-off*: la silhouette favorisce cluster sferici e ben
    separati. Per dataset più complessi può preferire K piccoli;
    in tal caso considerare BIC (per GMM) o davies-bouldin.
    """
    results = []
    for k in k_range:
        r = fit_fn(X, n_clusters=k, random_state=random_state)
        logger.info("  K=%d → silhouette=%.4f, inertia=%s", k, r.silhouette, r.inertia)
        results.append(r)
    best = max(results, key=lambda r: r.silhouette)
    logger.info("Best K=%d (silhouette=%.4f).", best.n_clusters, best.silhouette)
    return best.n_clusters, results


def _silhouette_sample(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
    random_state: int = RANDOM_STATE,
) -> float:
    """Silhouette su un sottocampione casuale (per scalabilità)."""
    n = len(X)
    if n <= sample_size:
        return float(silhouette_score(X, labels))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


__all__ = [
    "ClusteringResult",
    "fit_minibatch_kmeans",
    "fit_kmeans",
    "fit_gmm",
    "select_k_by_silhouette",
]
