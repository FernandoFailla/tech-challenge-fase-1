"""Tests for metrics module (src.training.metrics).

Tests the canonical implementation of compute_binary_classification_metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.metrics import compute_binary_classification_metrics


class TestComputeBinaryClassificationMetrics:
    """Test canonical implementation in src.training.metrics."""

    @staticmethod
    @pytest.mark.fast
    def test_perfect_predictions_numpy() -> None:
        """Should compute correct metrics with numpy arrays."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    @staticmethod
    @pytest.mark.fast
    def test_with_probabilities_numpy() -> None:
        """Should compute AUC metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.9])

        metrics = compute_binary_classification_metrics(y_true, y_pred, y_prob)

        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    @staticmethod
    @pytest.mark.fast
    def test_zero_division_handling() -> None:
        """Should handle zero division gracefully."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    @staticmethod
    @pytest.mark.fast
    def test_torch_tensor_inputs() -> None:
        """Should handle PyTorch tensors directly."""
        y_true = torch.tensor([1, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])
        y_prob = torch.tensor([0.9, 0.2, 0.3, 0.8])

        metrics = compute_binary_classification_metrics(
            y_true, y_pred, y_prob, positive_label=None
        )

        assert isinstance(metrics["accuracy"], float)
        assert "roc_auc" in metrics

    @staticmethod
    @pytest.mark.fast
    def test_pandas_series_with_string_labels() -> None:
        """Should handle pandas Series with string labels."""
        y_true = pd.Series(["Yes", "No", "Yes", "No"])
        y_pred = pd.Series(["Yes", "No", "No", "No"])
        y_prob = pd.Series([0.9, 0.1, 0.3, 0.2])

        metrics = compute_binary_classification_metrics(
            y_true, y_pred, y_prob, positive_label="Yes"
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "roc_auc" in metrics

    @staticmethod
    @pytest.mark.fast
    def test_numpy_arrays_without_positive_label() -> None:
        """Should work with numpy arrays without positive_label."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" not in metrics  # no probabilities

    @staticmethod
    @pytest.mark.fast
    def test_pandas_series_numeric_without_label() -> None:
        """Should work with numeric pandas Series without label."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_score" in metrics
