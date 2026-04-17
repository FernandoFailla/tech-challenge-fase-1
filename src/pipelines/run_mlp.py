"""Pipeline de treino para modelo MLP (Multi-Layer Perceptron).

Este script orquestra o treinamento do modelo MLP para predição de churn:
1. Carregamento dos dados brutos do dataset Telco Customer Churn
2. Pré-processamento (codificação, escalonamento)
3. Divisão treino/teste estratificada
4. Configuração e treino do modelo MLP
5. Avaliação no conjunto de teste
6. Logging de métricas e modelo no MLflow

Como usar:
    $ uv run python -m src.pipelines.train_mlp
    $ uv run python -m src.pipelines.train_mlp --input path/to/data.csv
    $ uv run python -m src.pipelines.train_mlp --experiment-name churn-mlp-v2

Requerimentos:
    - Arquivo .env configurado com MLFLOW_TRACKING_URI
    - Dados na estrutura esperada (veja load_telco_data)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from src.data.prepare_telco_dataset import load_telco_data
from src.data.preprocessing import mlp_preprocess_data
from src.data.splitting import split_train_test_stratified
from src.data.validation import validate_required_columns
from src.models import MLPConfig, MLPForTraining, TrainingConfig
from src.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from src.training import MLPTrainer
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

# Limiar para converter probabilidades em predições binárias
THRESHOLD: float = 0.5
RANDOM_SEED: int = 42
TARGET_COLUMN: str = "Churn"


def main() -> None:  # noqa: PLR0914, PLR0915
    """Função principal que executa o pipeline de treino completo.

    Orquestra todo o fluxo de ML:
    1. Parse de argumentos da linha de comando
    2. Carregamento e pré-processamento de dados
    3. Configuração do modelo e treinamento
    4. Avaliação no conjunto de teste
    5. Logging no MLflow

    Argumentos CLI:
        --input: Caminho para o dataset CSV
            (default: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)
        --experiment-name: Nome do experimento no MLflow

    Requerimentos de ambiente:
        - Arquivo .env com MLFLOW_TRACKING_URI
        - MLflow server rodando (iniciar com make docker-up)
    """
    # Configura argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Treina modelo MLP para predição de churn"
    )
    parser.add_argument(
        "--input",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Caminho para o dataset de entrada",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nome do experimento no MLflow",
    )
    args = parser.parse_args()

    # Carrega variáveis de ambiente (.env)
    load_dotenv_silent()

    # Configura MLflow via módulo genérico
    # Prioridade: CLI arg > MLP-specific env > generic env > default
    experiment_name = get_experiment_name(
        cli_arg=args.experiment_name,
        env_var_name="MLFLOW_MLP_EXPERIMENT_NAME",
        default_name="tech-challenge-mlp",
    )
    mlflow_config = MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=experiment_name,
    )
    setup_mlflow(mlflow_config)

    # === 1. CARREGAMENTO DE DADOS ===
    logger.info(f"Carregando dados de {args.input}")
    df = load_telco_data(args.input)

    # Validação de dados
    validate_required_columns(df, TARGET_COLUMN)

    # === 2. PRÉ-PROCESSAMENTO ===
    logger.info("Pré-processando dados")
    X, y, _feature_names = mlp_preprocess_data(df)

    # === 3. DIVISÃO TREINO/TESTE ===
    # Cria DataFrame temporário para usar split estratificado genérico
    df_processed = pd.DataFrame(X)
    df_processed[TARGET_COLUMN] = y

    X_train_df, X_test_df, y_train, y_test = split_train_test_stratified(
        df_processed,
        TARGET_COLUMN,
        test_size=0.2,
        random_seed=RANDOM_SEED,
    )

    # Converte de volta para numpy
    X_train = X_train_df.values
    X_test = X_test_df.values
    # Converte target de string para número (0.0 -> 0, 1.0 -> 1) para PyTorch
    y_train_numeric = pd.to_numeric(y_train, errors="coerce")
    y_test_numeric = pd.to_numeric(y_test, errors="coerce")
    # Verifica se há NaNs antes da conversão para numpy
    if y_train_numeric.isna().any() or y_test_numeric.isna().any():
        print("WARNING: NaN values found in target data")
        print(f"y_train NaN count: {y_train_numeric.isna().sum()}")
        print(f"y_test NaN count: {y_test_numeric.isna().sum()}")
        # Remove entradas com NaN
        valid_train_indices = y_train_numeric.notna()
        valid_test_indices = y_test_numeric.notna()
        X_train = X_train[valid_train_indices]
        X_test = X_test[valid_test_indices]
        y_train_numeric = y_train_numeric[valid_train_indices]
        y_test_numeric = y_test_numeric[valid_test_indices]

    # Converte para numpy arrays (tipado explícito para mypy)
    y_train_arr: np.ndarray = np.asarray(y_train_numeric.values)
    y_test_arr: np.ndarray = np.asarray(y_test_numeric.values)

    logger.info(f"Conjunto de treino: {X_train.shape[0]} amostras")
    logger.info(f"Conjunto de teste: {X_test.shape[0]} amostras")
    logger.info(f"Número de features: {X_train.shape[1]}")

    # Prepara lineage de dados para MLflow
    train_test_data = TrainTestData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train_arr,
        y_test=y_test_arr,
    )

    # Obtém versão do dataset via DVC
    dataset_version = safe_get_dataset_version()

    train_input, test_input = build_mlflow_inputs(
        train_test_data,
        TARGET_COLUMN,
        dataset_version,
        dataset_source_path=args.input,
    )

    # === 4. CONFIGURAÇÃO DO MODELO ===
    # Arquitetura: 3 camadas ocultas (128, 64, 32)
    # Dropout 0.3 para regularização
    # BatchNorm para estabilidade
    mlp_config = MLPConfig(
        input_dim=X_train.shape[1],
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
        use_batch_norm=True,
    )

    # Hiperparâmetros de treino
    # Adam otimizador com lr=0.001 (padrão que funciona bem)
    # Early stopping para prevenir overfitting
    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.001,
        weight_decay=1e-5,
        scheduler="reduce_on_plateau",
        scheduler_patience=3,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        batch_size=64,
        max_epochs=100,
        val_split=0.2,
        random_seed=42,
    )

    # === 5. TREINAMENTO COM MLFLOW ===
    with mlflow.start_run():
        # Log inputs de dados (type: ignore para mlflow typed stubs)
        mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
        mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]

        # Registra parâmetros da arquitetura
        mlflow.log_params(
            {
                "input_dim": mlp_config.input_dim,
                "hidden_dims": str(mlp_config.hidden_dims),
                "dropout_rate": mlp_config.dropout_rate,
                "use_batch_norm": mlp_config.use_batch_norm,
                "model_type": "MLP",
                "random_seed": RANDOM_SEED,
                "dataset_version": dataset_version,
            }
        )

        # Inicializa modelo e trainer
        model = MLPForTraining(mlp_config)
        trainer = MLPTrainer(model, training_config)

        # Treina com validação e early stopping
        logger.info("Iniciando treinamento")
        model_save_path = Path("models/churn_mlp_best.pt")
        _history = trainer.fit(
            X_train, y_train_arr, model_save_path=str(model_save_path)
        )

        # Registra métricas de treino no MLflow
        trainer.log_to_mlflow()

        logger.info("Treinamento concluído")

        # === 6. AVALIAÇÃO NO CONJUNTO DE TESTE ===
        # Coloca modelo em modo avaliação (desativa dropout)
        model.model.eval()
        # Desativa cálculo de gradientes (economiza memória, mais rápido)
        with torch.no_grad():
            # Converte dados de teste para tensor e move para GPU/CPU
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
                trainer.device
            )
            outputs = model(X_test_tensor)
            probs = outputs["probs"].cpu().numpy()
            # Converte probabilidades em predições binárias
            preds = (probs > THRESHOLD).astype(int)

        # Calcula métricas de classificação
        # positive_label=None indica dados já numéricos (0/1)
        test_metrics = compute_binary_classification_metrics(
            y_true=y_test_arr,
            y_pred=preds,
            y_proba_positive=probs,
            positive_label=None,
        )
        logger.info(f"Métricas de teste: {test_metrics}")

        # Registra métricas de teste no MLflow
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Salva modelo no MLflow registry
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"Modelo salvo em {model_save_path}")


if __name__ == "__main__":  # pragma: no cover
    # Configura logging básico ao nível INFO
    logging.basicConfig(level=logging.INFO)
    main()
