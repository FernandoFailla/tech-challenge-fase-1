from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.metrics import ClassificationMetrics


@pytest.mark.fast
def test_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])

    metrics = ClassificationMetrics.compute(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


@pytest.mark.fast
def test_with_probabilities() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.9])

    metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)

    assert "auc_roc" in metrics
    assert metrics["auc_roc"] == 1.0


@pytest.mark.fast
def test_zero_division_handling() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = ClassificationMetrics.compute(y_true, y_pred)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.fast
def test_torch_tensor_inputs() -> None:
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 1])
    y_prob = torch.tensor([0.1, 0.9, 0.2, 0.8])

    metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)

    assert isinstance(metrics["accuracy"], float)
    assert "auc_roc" in metrics
