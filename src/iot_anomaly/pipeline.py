"""Orchestratore end-to-end della pipeline IoT anomaly detection.

Esegue, nell'ordine:

1. Caricamento dataset (CSV → DataFrame ordinato per asset, time).
2. Wrangling: gestione missing per asset.
3. Feature engineering: rolling mean/std, diff, z-score.
4. Time-aware split: primi N giorni → train (assunti normali), resto → test.
5. Scaling robusto (StandardScaler).
6. (Opzionale) PCA per ridurre dimensionalità.
7. Selezione di K via silhouette.
8. Fit MiniBatchKMeans + soglia su percentile delle distanze train.
9. Scoring + predizione su test.
10. Valutazione vs ground-truth parziale (anomaly_label, fault_code_true).
11. Persistenza artefatti.

Eseguibile come modulo:

    iot-detect              # full run
    iot-detect --quick      # smoke test rapido
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .clustering import (
    fit_minibatch_kmeans,
    select_k_by_silhouette,
)
from .config import (
    DEFAULT_CONFIG,
    FIGURES_DIR,
    KMEANS_N_CLUSTERS_DEFAULT,
    KMEANS_N_CLUSTERS_RANGE,
    LABEL_COLUMNS,
    MODELS_DIR,
    REPORTS_DIR,
    PipelineConfig,
)
from .data import load_raw, time_split
from .evaluation import (
    evaluate,
    plot_pr_curve,
    plot_score_distribution,
)
from .features import TimeSeriesFeatureEngineer, select_modeling_features
from .scoring import fit_anomaly_detector
from .wrangling import add_missing_flags, fill_missing_per_asset

logger = logging.getLogger(__name__)


def run_full_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
) -> dict:
    """Esegue la pipeline completa e restituisce un dict di risultati.

    Args:
        config: configurazione (random_state, train_days, ecc.).
        quick: riduce K range per smoke-test (~30s).

    Returns:
        dict con: train/test DataFrame arricchiti, model, detector,
        metriche di evaluation, paths degli artefatti salvati.
    """
    logger.info("=" * 70)
    logger.info("Pipeline IoT anomaly detection (quick=%s)", quick)
    logger.info("=" * 70)

    # --- 1. Load ---
    df = load_raw()

    # --- 2. Wrangling ---
    # Gestisce sia colonne sensoriali che di contesto (load_pct, humidity_pct,
    # ambient_temp_c hanno anche loro buchi sparsi).
    from .config import SENSOR_COLUMNS
    cols_to_fill = SENSOR_COLUMNS + ("ambient_temp_c", "humidity_pct", "load_pct")
    df = add_missing_flags(df, columns=cols_to_fill)
    df = fill_missing_per_asset(df, columns=cols_to_fill)

    # --- 3. Feature engineering ---
    fe = TimeSeriesFeatureEngineer(window=config.rolling_window_min)
    df_fe = fe.fit_transform(df)

    # --- 4. Time-aware split ---
    train_df, test_df, cutoff = time_split(df_fe, config)

    modeling_features = select_modeling_features(df_fe)
    logger.info("Numero feature di modellazione: %d", len(modeling_features))

    X_train = train_df[modeling_features].to_numpy()
    X_test = test_df[modeling_features].to_numpy()

    # --- 5. Scaling ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- 6. PCA opzionale ---
    pca = None
    if config.use_pca:
        pca = PCA(n_components=config.pca_n_components, random_state=config.random_state)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
        logger.info("PCA: ridotto a %d componenti (varianza spiegata=%.3f).",
                    config.pca_n_components, float(pca.explained_variance_ratio_.sum()))

    # --- 7. Selezione K ---
    if config.kmeans_n_clusters is not None:
        best_k = config.kmeans_n_clusters
        logger.info("K fissato da config: %d", best_k)
    elif quick:
        best_k = KMEANS_N_CLUSTERS_DEFAULT
        logger.info("Quick mode: K=%d (default).", best_k)
    else:
        logger.info("Selezione di K via silhouette su K ∈ %s...", KMEANS_N_CLUSTERS_RANGE)
        best_k, _ = select_k_by_silhouette(
            X_train_s, k_range=KMEANS_N_CLUSTERS_RANGE,
            random_state=config.random_state,
        )

    # --- 8. Fit clustering + soglia ---
    cluster_result = fit_minibatch_kmeans(X_train_s, n_clusters=best_k,
                                           random_state=config.random_state)
    detector = fit_anomaly_detector(
        cluster_result.model, X_train_s,
        threshold_percentile=config.anomaly_threshold_percentile,
    )

    # --- 9. Scoring + predict ---
    train_scores = detector.score(X_train_s)
    test_scores = detector.score(X_test_s)
    train_pred = (train_scores >= detector.threshold).astype(int)
    test_pred = (test_scores >= detector.threshold).astype(int)

    train_df = train_df.assign(anomaly_score=train_scores, anomaly_pred=train_pred)
    test_df = test_df.assign(anomaly_score=test_scores, anomaly_pred=test_pred)

    # --- 10. Evaluation ---
    eval_train = evaluate(
        y_true_anomaly=train_df["anomaly_label"].to_numpy(),
        y_pred_anomaly=train_pred,
        y_score=train_scores,
        fault_code_true=train_df["fault_code_true"].to_numpy(),
    )
    eval_test = evaluate(
        y_true_anomaly=test_df["anomaly_label"].to_numpy(),
        y_pred_anomaly=test_pred,
        y_score=test_scores,
        fault_code_true=test_df["fault_code_true"].to_numpy(),
    )
    logger.info("\n=== Validation TRAIN ===")
    _log_eval(eval_train)
    logger.info("\n=== Validation TEST ===")
    _log_eval(eval_test)

    # --- 11. Save artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "anomaly_detector.joblib"
    joblib.dump({
        "feature_engineer": fe,
        "scaler": scaler,
        "pca": pca,
        "detector": detector,
        "modeling_features": modeling_features,
        "wrangling_columns": list(cols_to_fill),
        "config": config.__dict__,
    }, artifact_path)
    logger.info("Salvato: %s", artifact_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({
        "best_k": best_k,
        "silhouette": cluster_result.silhouette,
        "inertia": cluster_result.inertia,
        "threshold": detector.threshold,
        "threshold_percentile": detector.threshold_percentile,
        "cutoff": str(cutoff),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "validation_train": eval_train.as_dict(),
        "validation_test": eval_test.as_dict(),
    }, indent=2))

    # --- Plot diagnostici (best-effort) ---
    try:
        plot_score_distribution(
            train_scores, test_scores, detector.threshold,
            save_path=FIGURES_DIR / "score_distribution.png",
        )
        if test_df["anomaly_label"].sum() > 0:
            plot_pr_curve(
                test_df["anomaly_label"].to_numpy(), test_scores,
                save_path=FIGURES_DIR / "pr_curve_test.png",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot diagnostici saltati: %s", exc)

    return {
        "best_k": best_k,
        "detector": detector,
        "train_df": train_df,
        "test_df": test_df,
        "eval_train": eval_train,
        "eval_test": eval_test,
        "artifact_path": artifact_path,
        "metrics_path": metrics_path,
    }


def _log_eval(e) -> None:
    logger.info(
        "  precision=%.3f  recall=%.3f  f1=%.3f  ROC_AUC=%.3f  PR_AUC=%.3f",
        e.precision, e.recall, e.f1, e.roc_auc, e.pr_auc,
    )
    logger.info(
        "  predicted_anomalies=%d  true_anomalies=%d  fault_code_recall=%.3f",
        e.n_anomalies_predicted, e.n_anomalies_true, e.fault_code_recall,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoT Anomaly Detection — pipeline end-to-end.")
    parser.add_argument("--quick", action="store_true", help="Smoke-test rapido (K fisso).")
    parser.add_argument("--use-pca", action="store_true", help="Applica PCA prima del clustering.")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile per la soglia anomalia (default 99).")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()
    config = PipelineConfig(
        random_state=DEFAULT_CONFIG.random_state,
        train_days=DEFAULT_CONFIG.train_days,
        rolling_window_min=DEFAULT_CONFIG.rolling_window_min,
        kmeans_n_clusters=DEFAULT_CONFIG.kmeans_n_clusters,
        anomaly_threshold_percentile=args.threshold_percentile,
        use_pca=args.use_pca,
        pca_n_components=DEFAULT_CONFIG.pca_n_components,
        n_jobs=DEFAULT_CONFIG.n_jobs,
        verbose=DEFAULT_CONFIG.verbose,
    )
    run_full_pipeline(config=config, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_full_pipeline"]
