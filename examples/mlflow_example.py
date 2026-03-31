#!/usr/bin/env python3
<<<<<<< HEAD
"""Exemplo moderno de uso do MLflow com tracking server local.

Este script demonstra como usar o MLflow para tracking de experimentos
com código moderno, type hints e boas práticas.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.NullHandler()],
)
logger = logging.getLogger(__name__)

# Console Rich
console = Console()


class DatasetName(Enum):
    """Nomes dos datasets suportados."""

    IRIS = auto()
    WINE = auto()


class ModelName(Enum):
    """Nomes dos modelos suportados."""

    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"


@dataclass(frozen=True)
class ModelConfig:
    """Configuração de um modelo."""

    name: ModelName
    model_class: type[BaseEstimator]
    params: dict[str, Any]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuração de um dataset."""

    name: DatasetName
    data: NDArray[Any]
    target: NDArray[Any]
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ExperimentResult:
    """Resultado de uma execução de experimento."""

    run_id: str
    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    params: dict[str, Any]

    def to_table_row(self) -> list[str]:
        """Converte o resultado em uma linha de tabela."""
        return [
            self.model_name,
            self.dataset_name,
            f"{self.metrics['accuracy']:.4f}",
            f"{self.metrics['f1_score']:.4f}",
            self.run_id[:8] + "...",
        ]


class ModelTrainer(Protocol):
    """Protocolo para treinadores de modelo."""

    def train(
        self,
        model: BaseEstimator,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
    ) -> BaseEstimator:
        """Treina o modelo."""
        ...

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: NDArray[Any],
        y_test: NDArray[Any],
    ) -> dict[str, float]:
        """Avalia o modelo."""
        ...


@dataclass
class SklearnTrainer:
    """Treinador para modelos sklearn."""

    def train(
        self,
        model: BaseEstimator,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
    ) -> BaseEstimator:
        """Treina o modelo."""
        model.fit(X_train, y_train)
        return model

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: NDArray[Any],
        y_test: NDArray[Any],
    ) -> dict[str, float]:
        """Avalia o modelo e retorna métricas."""
        y_pred = model.predict(X_test)

        return {
=======
"""
Exemplo de uso do MLflow com tracking server local
Este script demonstra como usar o MLflow para tracking de experimentos
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# Carregar variáveis de ambiente (se existir .env)
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Variáveis de ambiente carregadas do .env")
except ImportError:
    print("⚠️  python-dotenv não instalado. Usando variáveis padrão.")

# Configuração do MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "tech-challenge-demo"
)

print(f"\n🔌 Conectando ao MLflow: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configurar S3/MinIO para artifacts (se estiver usando versão completa)
if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
        "AWS_ACCESS_KEY_ID", "minioadmin"
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY", "minioadmin"
    )
    print("✅ Configurado S3 endpoint para artifacts")

# Criar/Setar experimento
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"📊 Experimento: {MLFLOW_EXPERIMENT_NAME}\n")


def train_and_log_model(
    model, model_name, dataset_name, X_train, X_test, y_train, y_test, params
):
    """Treina um modelo e loga no MLflow"""

    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
        print(f"🚀 Iniciando run: {model_name} - {dataset_name}")

        # Log de parâmetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset", dataset_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Treinamento
        model.fit(X_train, y_train)

        # Predições
        y_pred = model.predict(X_test)

        # Métricas
        metrics = {
>>>>>>> c2c42ab (feat: implementa setup local de MLflow com PostgreSQL e MinIO (#45))
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

<<<<<<< HEAD

@dataclass
class MLflowConfig:
    """Configuração do MLflow."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tech-challenge-demo"
    s3_endpoint: str | None = None
    aws_access_key: str = "minioadmin"
    aws_secret_key: str = "minioadmin"

    @classmethod
    def from_env(cls) -> MLflowConfig:
        """Carrega configuração do ambiente."""
        load_dotenv()

        return cls(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            ),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "tech-challenge-demo"
            ),
            s3_endpoint=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )

    def setup(self) -> None:
        """Configura o MLflow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        if self.s3_endpoint:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_key
            console.print(
                "[green]✅ Configurado S3 endpoint para artifacts[/green]"
            )


class ExperimentRunner:
    """Executor de experimentos MLflow."""

    def __init__(
        self,
        config: MLflowConfig,
        trainer: ModelTrainer | None = None,
    ) -> None:
        """Inicializa o executor."""
        self.config = config
        self.trainer = trainer or SklearnTrainer()
        self.results: list[ExperimentResult] = []

    def run_experiment(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ) -> ExperimentResult:
        """Executa um experimento completo."""
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            dataset_config.data,
            dataset_config.target,
            test_size=dataset_config.test_size,
            random_state=dataset_config.random_state,
        )

        run_name = f"{model_config.name.value}_{dataset_config.name.name}"
        console.print(f"\n[blue]🚀 Iniciando run:[/blue] {run_name}")

        with mlflow.start_run(run_name=run_name):
            # Instanciar modelo
            model: BaseEstimator = model_config.model_class(
                **model_config.params
            )

            # Log de parâmetros
            mlflow.log_param("model_name", model_config.name.value)
            mlflow.log_param("dataset", dataset_config.name.name)
            for param_name, param_value in model_config.params.items():
                mlflow.log_param(param_name, param_value)

            # Treinamento
            trained_model = self.trainer.train(model, X_train, y_train)

            # Avaliação
            metrics = self.trainer.evaluate(trained_model, X_test, y_test)

            # Log de métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log de modelo
            mlflow.sklearn.log_model(trained_model, "model")

            # Coletar resultado
            run_id = mlflow.active_run().info.run_id
            result = ExperimentResult(
                run_id=run_id,
                model_name=model_config.name.value,
                dataset_name=dataset_config.name.name,
                metrics=metrics,
                params=model_config.params,
            )
            self.results.append(result)

            console.print(
                f"   [green]✅ Accuracy:[/green] {metrics['accuracy']:.4f}"
            )
            console.print(f"   [dim]Run ID: {run_id}[/dim]")

            return result

    def display_results(self) -> None:
        """Exibe os resultados em formato de tabela."""
        if not self.results:
            console.print("[yellow]⚠️ Nenhum resultado para exibir[/yellow]")
            return

        table = Table(
            title="📊 Resultados dos Experimentos",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Modelo", style="cyan")
        table.add_column("Dataset", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Run ID", style="dim")

        for result in self.results:
            table.add_row(*result.to_table_row())

        console.print(table)


def load_dataset(config: DatasetName) -> DatasetConfig:
    """Carrega um dataset pelo nome."""
    loaders = {
        DatasetName.IRIS: load_iris,
        DatasetName.WINE: load_wine,
    }

    data = loaders[config]()
    return DatasetConfig(
        name=config,
        data=data.data,
        target=data.target,
    )


def verify_mlflow_connection(config: MLflowConfig) -> bool:
    """Verifica a conexão com MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        console.print(f"[green]✅ Conexão com MLflow estabelecida![/green]")
        console.print(
            f"   [dim]Experiments disponíveis: {len(experiments)}[/dim]"
        )
        return True
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]❌ Erro ao conectar ao MLflow: {e}[/red]")
        console.print(
            "\n[yellow]💡 Certifique-se de que o servidor está rodando:[/yellow]"
        )
        console.print("   ./mlflow.sh start")
        return False


def main() -> int:
    """Função principal."""
    console.print(
        Panel.fit(
            "[bold blue]🧪 MLflow Experiments Demo[/bold blue]",
            border_style="blue",
        )
    )

    # Configuração
    config = MLflowConfig.from_env()
    console.print(
        f"\n[dim]🔌 Conectando ao MLflow:[/dim] {config.tracking_uri}"
    )

    config.setup()
    console.print(f"[dim]📊 Experimento:[/dim] {config.experiment_name}\n")

    # Verificar conexão
    if not verify_mlflow_connection(config):
        return 1

    # Configurações dos experimentos
    experiments: list[tuple[ModelConfig, DatasetConfig]] = [
        # Dataset Iris
        (
            ModelConfig(
                name=ModelName.RANDOM_FOREST,
                model_class=RandomForestClassifier,
                params={
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42,
                },
            ),
            load_dataset(DatasetName.IRIS),
        ),
        (
            ModelConfig(
                name=ModelName.GRADIENT_BOOSTING,
                model_class=GradientBoostingClassifier,
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
            ),
            load_dataset(DatasetName.IRIS),
        ),
        # Dataset Wine
        (
            ModelConfig(
                name=ModelName.RANDOM_FOREST,
                model_class=RandomForestClassifier,
                params={
                    "n_estimators": 200,
                    "max_depth": 10,
                    "random_state": 42,
                },
            ),
            load_dataset(DatasetName.WINE),
        ),
    ]

    # Executar experimentos
    runner = ExperimentRunner(config)

    console.print("\n" + "=" * 60)
    console.print("[bold]Iniciando experimentos...[/bold]")
    console.print("=" * 60)

    for model_config, dataset_config in experiments:
        runner.run_experiment(model_config, dataset_config)

    # Resultados
    console.print("\n" + "=" * 60)
    runner.display_results()

    # Instruções finais
    console.print("\n")
    console.print(
        Panel(
            Text(
                f"✅ Experimentos concluídos!\n\n"
                f"🔗 Acesse a interface do MLflow:\n"
                f"   {config.tracking_uri}\n\n"
                f"📊 Lá você pode:\n"
                f"   • Comparar métricas entre runs\n"
                f"   • Visualizar parâmetros\n"
                f"   • Baixar modelos treinados\n"
                f"   • Registrar modelos para produção",
                justify="left",
            ),
            title="Próximos Passos",
            border_style="green",
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
=======
        # Log de métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log de modelo
        mlflow.sklearn.log_model(model, "model")

        # Log de artifact (exemplo)
        # mlflow.log_artifact("caminho/para/arquivo.txt")

        print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✅ Run ID: {mlflow.active_run().info.run_id}")

        return metrics


def main():
    """Função principal - demonstração completa"""

    # Dataset 1: Iris
    print("=" * 60)
    print("📊 Dataset: Iris")
    print("=" * 60)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )

    # Experimento 1: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    train_and_log_model(
        rf_model,
        "RandomForest",
        "Iris",
        X_train_i,
        X_test_i,
        y_train_i,
        y_test_i,
        rf_params,
    )

    # Experimento 2: Gradient Boosting
    gb_params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
    gb_model = GradientBoostingClassifier(**gb_params)
    train_and_log_model(
        gb_model,
        "GradientBoosting",
        "Iris",
        X_train_i,
        X_test_i,
        y_train_i,
        y_test_i,
        gb_params,
    )

    # Dataset 2: Wine
    print("\n" + "=" * 60)
    print("📊 Dataset: Wine")
    print("=" * 60)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )

    # Experimento 3: Random Forest com diferentes parâmetros
    rf_params2 = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    rf_model2 = RandomForestClassifier(**rf_params2)
    train_and_log_model(
        rf_model2,
        "RandomForest",
        "Wine",
        X_train_w,
        X_test_w,
        y_train_w,
        y_test_w,
        rf_params2,
    )

    print("\n" + "=" * 60)
    print("✅ Experimentos concluídos!")
    print("=" * 60)
    print(f"\n🔗 Acesse a interface do MLflow:")
    print(f"   {MLFLOW_TRACKING_URI}")
    print(f"\n📊 Lá você pode:")
    print("   • Comparar métricas entre runs")
    print("   • Visualizar parâmetros")
    print("   • Baixar modelos treinados")
    print("   • Registrar modelos para produção")
    print("")


if __name__ == "__main__":
    # Verificar conexão com MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ Conexão com MLflow estabelecida!")
        print(f"   Experiments disponíveis: {len(experiments)}")
    except Exception as e:
        print(f"❌ Erro ao conectar ao MLflow: {e}")
        print(f"\n💡 Certifique-se de que o servidor está rodando:")
        print(f"   ./mlflow.sh start")
        exit(1)

    main()
>>>>>>> c2c42ab (feat: implementa setup local de MLflow com PostgreSQL e MinIO (#45))
