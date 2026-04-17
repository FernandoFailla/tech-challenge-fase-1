"""Tests for MLflow tracking module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)


@pytest.fixture
def sample_train_test_data() -> TrainTestData:
    """Create sample train/test data."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    X_test = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})
    y_train = pd.Series([0, 1, 0], name="target")
    y_test = pd.Series([1, 0], name="target")

    return TrainTestData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


class TestMLflowConfig:
    """Tests for MLflowConfig dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Should have sensible defaults."""
        config = MLflowConfig()
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "tech-challenge-default"

    @staticmethod
    def test_custom_values() -> None:
        """Should accept custom values."""
        config = MLflowConfig(
            tracking_uri="http://mlflow.example.com:5000",
            experiment_name="custom-experiment",
        )
        assert config.tracking_uri == "http://mlflow.example.com:5000"
        assert config.experiment_name == "custom-experiment"


class TestSetupMLflow:
    """Tests for setup_mlflow function."""

    @staticmethod
    @patch("src.training.mlflow_tracking.mlflow")
    def test_sets_tracking_uri_and_experiment(
        mock_mlflow: MagicMock,
    ) -> None:
        """Should configure tracking URI and experiment."""
        config = MLflowConfig(
            tracking_uri="http://custom:5000",
            experiment_name="test-exp",
        )
        setup_mlflow(config)

        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://custom:5000"
        )
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")


class TestBuildMLflowInputs:
    """Tests for build_mlflow_inputs function."""

    @staticmethod
    @patch("src.training.mlflow_tracking.from_pandas")
    def test_build_inputs_returns_tuple(
        mock_from_pandas: MagicMock,
        sample_train_test_data: TrainTestData,
    ) -> None:
        """Should return tuple of train and test inputs."""
        mock_from_pandas.side_effect = ["train_input", "test_input"]

        train_input, test_input = build_mlflow_inputs(
            sample_train_test_data,
            target_column="Churn",
            dataset_version="abc123def456",
        )

        assert train_input == "train_input"
        assert test_input == "test_input"

    @staticmethod
    @patch("src.training.mlflow_tracking.from_pandas")
    def test_build_inputs_truncates_version(
        mock_from_pandas: MagicMock,
        sample_train_test_data: TrainTestData,
    ) -> None:
        """Should use first 8 chars of version for names."""
        mock_from_pandas.side_effect = ["train_input", "test_input"]

        build_mlflow_inputs(
            sample_train_test_data,
            target_column="Churn",
            dataset_version="abc123def45678901234567890",
        )

        # Check that names use truncated version
        calls = mock_from_pandas.call_args_list
        assert "train_split_vabc123de" in calls[0].kwargs["name"]
        assert "test_split_vabc123de" in calls[1].kwargs["name"]

    @staticmethod
    @patch("src.training.mlflow_tracking.from_pandas")
    def test_build_inputs_includes_target_column(
        mock_from_pandas: MagicMock,
        sample_train_test_data: TrainTestData,
    ) -> None:
        """Should include target column in datasets."""
        mock_from_pandas.side_effect = ["train_input", "test_input"]

        build_mlflow_inputs(
            sample_train_test_data,
            target_column="Churn",
            dataset_version="abc123",
        )

        # Check first call has target column
        call_args = mock_from_pandas.call_args_list[0]
        df_arg = call_args.args[0]
        assert "Churn" in df_arg.columns
