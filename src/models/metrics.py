"""Cálculo de métricas de classificação para predição binária.

Este módulo fornece funções utilitárias para computar métricas
padrão de classificação usando scikit-learn, com conversão
automática de tensores PyTorch. Projetado para funcionar
perfeitamente com loops de treino PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationMetrics:
    """Classe utilitária para cálculo de métricas de classificação binária.

    Esta classe fornece um método estático para computar todas as
    métricas relevantes para problemas de classificação binária. Ela
    lida com a conversão de tensores PyTorch para arrays numpy
    automaticamente, facilitando o uso em loops de treino.

    As métricas calculadas são:
    - Acurácia: Corretude geral (VP + VN) / (VP + VN + FP + FN)
    - Precisão: Valor preditivo positivo VP / (VP + FP)
    - Recall (Sensibilidade): Taxa de verdadeiros positivos
    - F1 Score: Média harmônica de precisão e recall
    - AUC-ROC: Área sob a curva ROC, mede qualidade do ranqueamento

    Para conjuntos desbalanceados (comum em churn), F1 e AUC-ROC
    são geralmente mais informativos que a acurácia isolada.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calcula métricas a partir de predições e valores reais.

        Lida automaticamente com a conversão de tensores PyTorch
        para arrays numpy, permitindo integração perfeita com
        saídas do modelo durante avaliação.

        Args:
            y_true: Rótulos verdadeiros (0 ou 1). Pode ser array
                numpy ou tensor PyTorch.
            y_pred: Rótulos de classe preditos (0 ou 1). Pode ser
                array numpy ou tensor PyTorch.
            y_prob: Probabilidades preditas para a classe positiva
                (classe 1). Requerido para cálculo do AUC-ROC.
                Pode ser None se AUC não for necessário.

        Returns:
            Dicionário contendo métricas calculadas:
                - accuracy: Acurácia geral de classificação
                - precision: Valor preditivo positivo (VPP)
                - recall: Taxa de verdadeiros positivos
                - f1: F1 score (média harmônica)
                - auc_roc: Área sob a curva ROC (se y_prob)

        Note:
            - zero_division=0 previne erros quando todas as
              predições são negativas
            - AUC-ROC retorna 0.0 se houver apenas uma classe

        Exemplo:
            >>> import torch
            >>> y_true = torch.tensor([1, 0, 1, 1])
            >>> y_pred = torch.tensor([1, 0, 0, 1])
            >>> y_prob = torch.tensor([0.9, 0.2, 0.3, 0.8])
            >>> metrics = ClassificationMetrics.compute(
            ...     y_true, y_pred, y_prob
            ... )
            >>> print(metrics['f1'])
            0.75
        """
        # Lida com tensores PyTorch convertendo para numpy
        # Permite integração com saídas do modelo durante avaliação
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.detach().cpu().numpy()

        # Calcula métricas base usando sklearn
        # zero_division=0 previne erros com predições de uma classe
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0.0),
            "recall": recall_score(y_true, y_pred, zero_division=0.0),
            "f1": f1_score(y_true, y_pred, zero_division=0.0),
        }

        # AUC-ROC requer probabilidades, não predições de classe
        # Mede quão bem o modelo ranqueia amostras positivas
        if y_prob is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                # AUC indefinido quando y_true tem apenas uma classe
                # Retorna 0.0 como fallback
                metrics["auc_roc"] = 0.0

        return metrics
