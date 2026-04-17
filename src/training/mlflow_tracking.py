"""Utilitários genéricos para tracking com MLflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
from mlflow.data import from_pandas  # type: ignore[attr-defined]

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_DATASET_SOURCE_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class MLflowConfig:
    """Configuração de tracking do MLflow."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tech-challenge-default"


@dataclass(frozen=True)
class TrainTestData:
    """Estrutura com split de treino e teste para lineage."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def setup_mlflow(config: MLflowConfig) -> None:
    """Configura tracking URI e experimento no MLflow."""
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    tracking_uri = config.tracking_uri
    if "localhost:5000" in tracking_uri:
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        )
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")


def build_mlflow_inputs(
    train_test_data: TrainTestData,
    target_column: str,
    dataset_version: str,
    dataset_source_path: str = DEFAULT_DATASET_SOURCE_PATH,
) -> tuple[object, object]:
    """Cria datasets de input para lineage sem digest fixo manual."""
    dataset_version_short = dataset_version[:8]

    train_dataset = train_test_data.X_train.copy()
    train_dataset[target_column] = train_test_data.y_train
    test_dataset = train_test_data.X_test.copy()
    test_dataset[target_column] = train_test_data.y_test

    train_input = from_pandas(
        train_dataset,
        source=dataset_source_path,
        name=f"train_split_v{dataset_version_short}",
    )
    test_input = from_pandas(
        test_dataset,
        source=dataset_source_path,
        name=f"test_split_v{dataset_version_short}",
    )
    return train_input, test_input
