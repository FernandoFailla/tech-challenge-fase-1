"""Métricas genéricas para classificação binária."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba_positive: pd.Series,
    positive_label: str,
) -> dict[str, float]:
    """Calcula métricas para classificação binária."""
    y_true_bin = (y_true == positive_label).astype(int)
    y_pred_bin = (y_pred == positive_label).astype(int)

    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1_score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, y_proba_positive),
        "pr_auc": average_precision_score(y_true_bin, y_proba_positive),
    }
