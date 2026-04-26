"""Métricas genéricas para classificação binária.

Este módulo fornece funções utilitárias para computar métricas
padrão de classificação binária, suportando pandas Series,
arrays numpy e tensores PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    import torch


def _to_numpy_array(
    data: pd.Series | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Converte Series pandas ou array para numpy array.

    Args:
        data: Series pandas, array numpy ou tensor PyTorch

    Returns:
        Array numpy unidimensional

    Raises:
        TypeError: Se o tipo não for suportado
    """
    if isinstance(data, pd.Series):
        return np.asarray(data.values)
    if isinstance(data, np.ndarray):
        return data
    # Lazy import para suportar torch sem dependência obrigatória
    try:
        # nosec: B404 - lazy import necessário para evitar dependência
        import torch  # noqa: PLC0415

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    raise TypeError(
        f"Tipo não suportado: {type(data)}. "
        "Esperado pd.Series, np.ndarray ou torch.Tensor"
    )


def compute_binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray | None = None,
    positive_label: str | None = None,
) -> dict[str, float]:
    """Calcula métricas para classificação binária.

    Esta função suporta múltiplos formatos de entrada:
    - pandas Series (requer positive_label para converter strings)
    - numpy arrays (assume valores já numéricos 0/1)
    - tensores PyTorch (converte automaticamente)

    Args:
        y_true: Rótulos verdadeiros. Se Series, deve conter strings
            como "Yes"/"No" com positive_label especificado, ou valores
            numéricos 0/1.
        y_pred: Rótulos preditos. Mesmo formato que y_true ou int.
        y_proba_positive: Probabilidades para classe positiva (0-1).
            Opcional, mas requerido para métricas AUC.
        positive_label: Label da classe positiva quando y_true/y_pred
            são Series de strings. Opcional se dados já numéricos.

    Returns:
        Dicionário com métricas:
            - accuracy: Acurácia geral
            - precision: Precisão (VPP)
            - recall: Recall (sensibilidade)
            - f1_score: F1 score
            - roc_auc: Área sob curva ROC (se y_proba_positive)
            - pr_auc: Área sob curva PR (se y_proba_positive)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Com pandas Series e labels string
        >>> y_true = pd.Series(["Yes", "No", "Yes"])
        >>> y_pred = pd.Series(["Yes", "No", "No"])
        >>> y_prob = pd.Series([0.9, 0.1, 0.3])
        >>> metrics = compute_binary_classification_metrics(
        ...     y_true, y_pred, y_prob, positive_label="Yes"
        ... )
        >>> # Com numpy arrays numéricos
        >>> y_true = np.array([1, 0, 1])
        >>> y_pred = np.array([1, 0, 0])
        >>> metrics = compute_binary_classification_metrics(
        ...     y_true, y_pred
        ... )
    """
    # Converte para numpy arrays
    y_true_arr = _to_numpy_array(y_true)
    y_pred_arr = _to_numpy_array(y_pred)

    # Converte labels string para binário se necessário
    if positive_label is not None:
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
    else:
        # Assume já numérico (0/1)
        y_true_bin = y_true_arr.astype(int)
        y_pred_bin = y_pred_arr.astype(int)

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1_score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }

    # Adiciona métricas baseadas em probabilidades se disponíveis
    if y_proba_positive is not None:
        y_proba_arr = _to_numpy_array(y_proba_positive)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_bin, y_proba_arr)
            metrics["pr_auc"] = average_precision_score(
                y_true_bin, y_proba_arr
            )
        except ValueError:
            # AUC indefinido quando há apenas uma classe
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    return metrics
