"""Pipeline completo de treino para predição de churn em telecomunicações.

Este script orquestra o fluxo completo de ML:
1. Carregamento dos dados brutos do dataset Telco Customer Churn
2. Pré-processamento (codificação, escalonamento)
3. Divisão treino/teste estratificada
4. Configuração e treino do modelo MLP
5. Avaliação no conjunto de teste
6. Logging de métricas e modelo no MLflow

Como usar:
    $ uv run python -m src.pipelines.train
    $ uv run python -m src.pipelines.train --input path/to/data.csv
    $ uv run python -m src.pipelines.train --experiment-name churn-mlp-v2

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
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.prepare_telco_dataset import load_telco_data
from src.models import MLPConfig, MLPForTraining, TrainingConfig
from src.models.metrics import ClassificationMetrics
from src.training import Trainer

logger = logging.getLogger(__name__)

# Limiar para converter probabilidades em predições binárias
THRESHOLD: float = 0.5


def preprocess_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pré-processa o DataFrame bruto do Telco para treino de ML.

    Esta função realiza transformações necessárias para preparar dados
    tabulares para uma rede neural:

    Passos de pré-processamento:
        1. Remove customerID (não é uma feature)
        2. Codifica Churn: "Yes"->1, "No"->0 (target binário)
        3. Converte TotalCharges para numérico, preenchendo NaNs com 0
        4. One-hot encoding para variáveis categóricas
        5. StandardScaler para variáveis numéricas (média=0, std=1)

    Por que cada passo:
        - One-hot: Redes neurais requerem entrada numérica
        - StandardScaler: Normalização acelera convergência
            e melhora estabilidade
        - drop_first=True: Evita multicolinearidade

    Args:
        df: DataFrame bruto do dataset Telco Customer Churn

    Returns:
        Tupla contendo:
            - X: Array de features de shape (n_samples,
                n_features)
            - y: Array de targets de shape (n_samples,)
            - feature_names: Lista com nomes das features
                após pré-processamento

    Exemplo:
        >>> df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        >>> X, y, features = preprocess_data(df)
        >>> print(f"Features: {len(features)}")  # ~45 após one-hot
    """
    # Remove ID do cliente - não é uma feature útil para predição
    df = df.drop(columns=["customerID"])

    # Codifica target: 1 para churn (Yes), 0 para retenção (No)
    # Usamos map para conversão explícita
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # TotalCharges às vezes vem como string vazia " " - converte para numérico
    # errors='coerce' converte valores inválidos em NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Preenche NaNs com 0 - assumindo clientes novos sem histórico de cobrança
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Separa colunas categóricas (object/string) e numéricas (int/float)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    numerical_cols.remove("Churn")  # Target não é feature

    # One-hot encoding: converte categorias em colunas binárias
    # drop_first=True remove uma categoria para evitar multicolinearidade
    # Exemplo: "Gender" -> "Gender_Male" (1=masculino, 0=feminino)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # StandardScaler: z = (x - média) / desvio_padrão
    # Importante para redes neurais - features na mesma escada evitam
    # gradientes dominantes e aceleram convergência
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(
        df_encoded[numerical_cols]
    )

    # Separa features (X) e target (y)
    # Converte para numpy arrays - formato esperado por PyTorch
    y = np.asarray(df_encoded["Churn"].values, dtype=np.float64)
    X = np.asarray(df_encoded.drop(columns=["Churn"]).values, dtype=np.float64)
    feature_names = df_encoded.drop(columns=["Churn"]).columns.tolist()

    return X, y, feature_names


def main() -> None:  # noqa: PLR0914
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
    load_dotenv()

    # Configura MLflow experiment
    experiment_name = args.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "tech-challenge-default"
    )
    mlflow.set_experiment(experiment_name)

    # === 1. CARREGAMENTO DE DADOS ===
    logger.info(f"Carregando dados de {args.input}")
    df = load_telco_data(args.input)

    # === 2. PRÉ-PROCESSAMENTO ===
    logger.info("Pré-processando dados")
    X, y, _feature_names = preprocess_data(df)

    # === 3. DIVISÃO TREINO/TESTE ===
    # stratify=y garante mesma proporção de churn em treino e teste
    # Importante para datasets desbalanceados (churn ~26% no Telco)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Conjunto de treino: {X_train.shape[0]} amostras")
    logger.info(f"Conjunto de teste: {X_test.shape[0]} amostras")
    logger.info(f"Número de features: {X_train.shape[1]}")

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
        # Registra parâmetros da arquitetura
        mlflow.log_params(
            {
                "input_dim": mlp_config.input_dim,
                "hidden_dims": str(mlp_config.hidden_dims),
                "dropout_rate": mlp_config.dropout_rate,
                "use_batch_norm": mlp_config.use_batch_norm,
                "model_type": "MLP",
            }
        )

        # Inicializa modelo e trainer
        model = MLPForTraining(mlp_config)
        trainer = Trainer(model, training_config)

        # Treina com validação e early stopping
        logger.info("Iniciando treinamento")
        model_save_path = Path("models/churn_mlp_best.pt")
        _history = trainer.fit(
            X_train, y_train, model_save_path=str(model_save_path)
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
        test_metrics = ClassificationMetrics.compute(y_test, preds, probs)
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
