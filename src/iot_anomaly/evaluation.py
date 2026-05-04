"""Validazione delle anomalie rilevate vs ground-truth parziale.

**Punto critico didattico**: le label `anomaly_label` e
`fault_code_true` NON sono usate per il training. Servono solo per
valutare A POSTERIORI quanto il clustering ha catturato le anomalie
*dichiarate*. Sono una validazione **debole** — il modello potrebbe
trovare anomalie reali NON labellate (false positive apparenti) o
mancare anomalie sottili (false negative).

Metriche calcolate:
- Precision/Recall/F1 vs `anomaly_label`.
- ROC-AUC e PR-AUC sulla distribuzione di score (continui).
- Detection rate dei `fault_code_true != 0`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvaluation:
    """Metriche di anomaly detection vs ground-truth parziale."""
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_anomalies_predicted: int
    n_anomalies_true: int
    fault_code_recall: float           # frazione di fault_code!=0 catturati come anomalia
    confusion_matrix: np.ndarray       # 2x2

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "n_anomalies_predicted": int(self.n_anomalies_predicted),
            "n_anomalies_true": int(self.n_anomalies_true),
            "fault_code_recall": self.fault_code_recall,
            "tn": int(self.confusion_matrix[0, 0]),
            "fp": int(self.confusion_matrix[0, 1]),
            "fn": int(self.confusion_matrix[1, 0]),
            "tp": int(self.confusion_matrix[1, 1]),
        }


def evaluate(
    y_true_anomaly: np.ndarray,
    y_pred_anomaly: np.ndarray,
    y_score: np.ndarray,
    fault_code_true: np.ndarray | None = None,
) -> AnomalyEvaluation:
    """Calcola tutte le metriche di anomaly detection."""
    y_true = np.asarray(y_true_anomaly).astype(int)
    y_pred = np.asarray(y_pred_anomaly).astype(int)
    y_sc = np.asarray(y_score, dtype=float)

    # Edge case: nessun positivo nel ground-truth.
    if y_true.sum() == 0:
        return AnomalyEvaluation(
            precision=float("nan"), recall=float("nan"), f1=float("nan"),
            roc_auc=float("nan"), pr_auc=float("nan"),
            n_anomalies_predicted=int(y_pred.sum()),
            n_anomalies_true=0,
            fault_code_recall=float("nan"),
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        )

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_sc)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_sc)
    except ValueError:
        pr = float("nan")

    fault_recall = float("nan")
    if fault_code_true is not None:
        is_fault = (np.asarray(fault_code_true) != 0).astype(int)
        if is_fault.sum() > 0:
            fault_recall = float((y_pred[is_fault.astype(bool)] == 1).mean())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return AnomalyEvaluation(
        precision=float(p), recall=float(r), f1=float(f1),
        roc_auc=float(roc), pr_auc=float(pr),
        n_anomalies_predicted=int(y_pred.sum()),
        n_anomalies_true=int(y_true.sum()),
        fault_code_recall=fault_recall,
        confusion_matrix=cm,
    )


# --- Plotting ---

def plot_score_distribution(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Distribuzione score train/test + linea soglia."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_scores, bins=80, alpha=0.6, label=f"train (n={len(train_scores)})", density=True)
    ax.hist(test_scores, bins=80, alpha=0.6, label=f"test (n={len(test_scores)})", density=True)
    ax.axvline(threshold, color="red", linestyle="--", lw=1.5, label=f"soglia={threshold:.3f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Densità")
    ax.set_title("Distribuzione dello score di anomalia")
    ax.legend()
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_temporal_anomalies(
    df: pd.DataFrame,
    anomaly_pred_col: str = "anomaly_pred",
    timestamp_col: str = "timestamp",
    asset_col: str = "asset_id",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap temporale: ogni riga = un asset, x = tempo, colore = anomalo o no."""
    pivot = df.pivot_table(
        index=asset_col,
        columns=timestamp_col,
        values=anomaly_pred_col,
        aggfunc="max",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(pivot.values, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(f"Tempo (n={pivot.shape[1]} timestamps)")
    ax.set_ylabel("asset_id")
    ax.set_title("Mappa anomalie nel tempo (rosso = anomalia)")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_pr_curve(
    y_true: np.ndarray, y_score: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Precision-Recall curve."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r, p, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve (AP = {ap:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


__all__ = [
    "AnomalyEvaluation",
    "evaluate",
    "plot_score_distribution",
    "plot_temporal_anomalies",
    "plot_pr_curve",
]
