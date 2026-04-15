"""Pipeline do baseline DummyClassifier para churn.

Fase 2: treino, métricas e registro no MLflow.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.prepare_telco_dataset import load_telco_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

RANDOM_SEED = 42
TARGET_COLUMN = "Churn"
STRATEGIES = ("most_frequent", "stratified", "uniform")
POSITIVE_LABEL = "Yes"
MIN_TARGET_CLASSES = 2


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração do pipeline baseline Dummy."""

    test_size: float = 0.2
    random_seed: int = RANDOM_SEED
    target_column: str = TARGET_COLUMN


@dataclass(frozen=True)
class MLflowConfig:
    """Configuração de tracking do MLflow."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tech-challenge-dummy-baseline"


TrainData = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]


def load_dataframe() -> pd.DataFrame:
    """Carrega o dataset base de churn."""
    return load_telco_data()


def validate_required_columns(df: pd.DataFrame, target_column: str) -> None:
    """Valida se a coluna alvo existe no dataset."""
    if df.empty:
        raise ValueError("Dataset vazio. Não é possível treinar baseline.")

    if target_column not in df.columns:
        msg = f"Coluna alvo ausente no dataset: '{target_column}'."
        raise ValueError(msg)


def validate_target_values(y: pd.Series) -> None:
    """Valida presença de classes esperadas na variável alvo."""
    unique_values = set(y.astype(str).unique().tolist())
    if POSITIVE_LABEL not in unique_values:
        msg = (
            "Classe positiva 'Yes' não encontrada na coluna alvo. "
            "Verifique o mapeamento de churn."
        )
        raise ValueError(msg)

    if len(unique_values) < MIN_TARGET_CLASSES:
        raise ValueError("A coluna alvo precisa ter pelo menos duas classes.")


def split_data(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste com seed fixa para reprodutibilidade."""
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba_positive: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de classificação binária para churn."""
    y_true_bin = (y_true == POSITIVE_LABEL).astype(int)
    y_pred_bin = (y_pred == POSITIVE_LABEL).astype(int)

    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1_score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, y_proba_positive),
        "pr_auc": average_precision_score(y_true_bin, y_proba_positive),
    }


def setup_mlflow(config: MLflowConfig) -> None:
    """Configura tracking URI e experimento no MLflow."""
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    tracking_uri = config.tracking_uri
    if "localhost:5000" in tracking_uri:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")


def run_all_strategies(
    train_data: TrainData,
    pipeline_config: PipelineConfig,
    mlflow_config: MLflowConfig,
) -> pd.DataFrame:
    """Executa treino/eval/log para cada estratégia do DummyClassifier.

    Retorna dataframe comparativo com métricas por estratégia.
    """
    setup_mlflow(mlflow_config)
    X_train, X_test, y_train, y_test = train_data
    results: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        model = DummyClassifier(
            strategy=strategy,
            random_state=pipeline_config.random_seed,
        )
        model.fit(X_train, y_train)

        y_pred = pd.Series(
            model.predict(X_test),
            index=y_test.index,
        ).astype(str)
        proba_classes = list(model.classes_)
        positive_idx = proba_classes.index(POSITIVE_LABEL)
        y_proba_positive = pd.Series(
            model.predict_proba(X_test)[:, positive_idx],
            index=y_test.index,
        )

        metrics = compute_metrics(y_test, y_pred, y_proba_positive)

        run_name = f"dummy_{strategy}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", "DummyClassifier")
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("random_seed", pipeline_config.random_seed)
            mlflow.log_param("test_size", pipeline_config.test_size)
            mlflow.log_param("target_column", pipeline_config.target_column)

            mlflow.set_tag("issue", "20")
            mlflow.set_tag("baseline_family", "dummy")
            mlflow.set_tag("model_baseline", "dummy_classifier")
            mlflow.set_tag("random_seed", str(pipeline_config.random_seed))

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        result = {"strategy": strategy, **metrics}
        results.append(result)
        print(
            "[Fase 2] Estratégia "
            f"{strategy}: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False,
    )
    output_path = Path("models/dummy_baseline_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    with mlflow.start_run(run_name="dummy_comparison_summary"):
        mlflow.log_param("model_type", "DummyClassifier")
        mlflow.log_param("random_seed", pipeline_config.random_seed)
        mlflow.log_param("strategies", ",".join(STRATEGIES))
        mlflow.set_tag("issue", "20")
        mlflow.set_tag("baseline_family", "dummy")
        mlflow.set_tag("model_baseline", "dummy_classifier")
        mlflow.set_tag("random_seed", str(pipeline_config.random_seed))
        mlflow.log_artifact(str(output_path))

    return results_df


def main() -> int:
    """Ponto de entrada do script."""
    config = PipelineConfig()
    mlflow_config = MLflowConfig()

    df = load_dataframe()
    validate_required_columns(df, config.target_column)
    validate_target_values(df[config.target_column])

    X_train, X_test, y_train, y_test = split_data(df, config)
    train_data = (X_train, X_test, y_train, y_test)
    results_df = run_all_strategies(train_data, config, mlflow_config)

    print("[Fase 2] Treino/eval/log no MLflow concluídos com sucesso.")
    print(
        "[Fase 2] Comparativo salvo em: "
        "models/dummy_baseline_comparison.csv"
    )
    print(results_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
