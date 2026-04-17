"""Pré-processamento de dados para modelos MLP.

Este módulo fornece funções de pré-processamento específicas
para preparação de dados tabulares para redes neurais.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def mlp_preprocess_data(
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

    Example:
        >>> df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        >>> X, y, features = mlp_preprocess_data(df)
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
    scaler: BaseEstimator = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(
        df_encoded[numerical_cols]
    )

    # Separa features (X) e target (y)
    # Converte para numpy arrays - formato esperado por PyTorch
    y = np.asarray(df_encoded["Churn"].values, dtype=np.float64)
    X = np.asarray(df_encoded.drop(columns=["Churn"]).values, dtype=np.float64)
    feature_names = df_encoded.drop(columns=["Churn"]).columns.tolist()

    return X, y, feature_names
