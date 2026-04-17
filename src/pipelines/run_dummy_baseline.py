"""Pipeline do baseline DummyClassifier para churn.

Fase 2: treino, metricas e registro no MLflow.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.constants import (
    DEFAULT_DUMMY_EXPERIMENT_NAME,
    POSITIVE_LABEL,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.data.prepare_telco_dataset import load_telco_data
from src.data.splitting import split_train_test_stratified
from src.data.validation import (
    validate_binary_target,
    validate_required_columns,
)
from src.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

STRATEGIES = ("most_frequent", "stratified", "uniform")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração do pipeline baseline Dummy."""

    test_size: float = 0.2
    random_seed: int = RANDOM_SEED
    target_column: str = TARGET_COLUMN


TrainData = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]


def run_all_strategies(  # noqa: PLR0914, PLR0915
    train_data: TrainData,
    pipeline_config: PipelineConfig,
    mlflow_config: MLflowConfig,
) -> pd.DataFrame:
    """Executa treino/eval/log para cada estratégia do DummyClassifier.

    Retorna dataframe comparativo com métricas por estratégia.
    """
    setup_mlflow(mlflow_config)
    X_train, X_test, y_train, y_test = train_data
    dataset_version = safe_get_dataset_version()
    train_test_data = TrainTestData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    train_input, test_input = build_mlflow_inputs(
        train_test_data,
        pipeline_config.target_column,
        dataset_version,
    )

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

        metrics = compute_binary_classification_metrics(
            y_test,
            y_pred,
            y_proba_positive,
            POSITIVE_LABEL,
        )

        run_name = f"dummy_{strategy}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
            mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]

            mlflow.log_param("model_type", "DummyClassifier")
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("random_seed", pipeline_config.random_seed)
            mlflow.log_param("test_size", pipeline_config.test_size)
            mlflow.log_param("target_column", pipeline_config.target_column)
            mlflow.log_param("dataset_version", dataset_version)

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

    best_row = results_df.iloc[0]
    output_path = Path("models/dummy_baseline_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    with mlflow.start_run(run_name="dummy_comparison_summary"):
        mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
        mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]

        mlflow.log_param("model_type", "DummyClassifier")
        mlflow.log_param("random_seed", pipeline_config.random_seed)
        mlflow.log_param("strategies", ",".join(STRATEGIES))
        mlflow.log_param("best_strategy", str(best_row["strategy"]))
        mlflow.log_param("dataset_version", dataset_version)

        mlflow.log_metric("best_f1_score", float(best_row["f1_score"]))
        mlflow.log_metric("best_accuracy", float(best_row["accuracy"]))

        mlflow.set_tag("issue", "20")
        mlflow.set_tag("baseline_family", "dummy")
        mlflow.set_tag("model_baseline", "dummy_classifier")
        mlflow.set_tag("random_seed", str(pipeline_config.random_seed))
        mlflow.log_artifact(str(output_path))

    return results_df


def main() -> int:
    """Ponto de entrada do script."""
    # Carrega variáveis de ambiente
    load_dotenv_silent()

    config = PipelineConfig()

    # Obtém nome do experimento com prioridade
    experiment_name = get_experiment_name(
        cli_arg=None,
        env_var_name="MLFLOW_DUMMY_EXPERIMENT_NAME",
        default_name=DEFAULT_DUMMY_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=experiment_name)

    df = load_telco_data()
    validate_required_columns(df, config.target_column)
    validate_binary_target(df[config.target_column], POSITIVE_LABEL)

    X_train, X_test, y_train, y_test = split_train_test_stratified(
        df,
        config.target_column,
        config.test_size,
        config.random_seed,
    )
    train_data = (X_train, X_test, y_train, y_test)
    results_df = run_all_strategies(train_data, config, mlflow_config)

    print("[Fase 2] Treino/eval/log no MLflow concluídos com sucesso.")
    print(
        "[Fase 2] Comparativo salvo em: models/dummy_baseline_comparison.csv"
    )
    print(results_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
