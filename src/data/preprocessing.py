"""Preprocessamento de dados para modelos MLP.

Este modulo fornece funcoes de preprocessamento especificas
para preparacao de dados tabulares para redes neurais.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def mlp_preprocess_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Preprocessa o DataFrame bruto do Telco para treino de ML.

    Esta funcao realiza transformacoes necessarias para preparar dados
    tabulares para uma rede neural. NAO aplica StandardScaler - isso
    deve ser feito APOS o split treino/teste para evitar data leakage.

    Passos de preprocessamento:
        1. Remove customerID (nao e uma feature)
        2. Codifica Churn: "Yes"->1, "No"->0 (target binario)
        3. Converte TotalCharges para numerico, preenchendo NaNs com 0
        4. One-hot encoding para variaveis categoricas
        5. Retorna features nao escaladas (scaling feito separadamente)

    Por que cada passo:
        - One-hot: Redes neurais requerem entrada numerica
        - Sem StandardScaler aqui: Evita data leakage - scaler deve ser
          fitado apenas no conjunto de treino
        - drop_first=True: Evita multicolinearidade

    Args:
        df: DataFrame bruto do dataset Telco Customer Churn

    Returns:
        Tupla contendo:
            - X: Array de features de shape (n_samples, n_features)
                 sem scaling aplicado
            - y: Array de targets de shape (n_samples,)
            - feature_names: Lista com nomes das features
                apos one-hot encoding
            - df_encoded: DataFrame com features codificadas (sem scaling)

    Example:
        >>> df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        >>> X, y, features, df_enc = mlp_preprocess_data(df)
        >>> print(f"Features: {len(features)}")  # ~45 apos one-hot
    """
    # Remove ID do cliente - nao e uma feature util para predicao
    df = df.drop(columns=["customerID"])

    # Codifica target: 1 para churn (Yes), 0 para retencao (No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # TotalCharges as vezes vem como string vazia " " - converte para numerico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Preenche NaNs com 0 - assumindo clientes novos sem historico de cobranca
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Separa colunas categoricas e numericas
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    numerical_cols.remove("Churn")  # Target nao e feature

    # One-hot encoding: converte categorias em colunas binarias
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separa features (X) e target (y)
    y = np.asarray(df_encoded["Churn"].values, dtype=np.float64)
    X = np.asarray(df_encoded.drop(columns=["Churn"]).values, dtype=np.float64)
    feature_names = df_encoded.drop(columns=["Churn"]).columns.tolist()

    return X, y, feature_names, df_encoded


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Cria e fita um StandardScaler nos dados de treino.

    IMPORTANTE: Sempre fit apenas no conjunto de treino para
    evitar data leakage. Use transform() nos dados de teste.

    Args:
        X_train: Features de treino de shape (n_samples, n_features)

    Returns:
        StandardScaler fitado nos dados de treino
    """
    scaler: BaseEstimator = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaling(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Aplica scaling em features usando scaler pre-fitado.

    Args:
        X: Array de features de shape (n_samples, n_features)
        scaler: StandardScaler ja fitado nos dados de treino

    Returns:
        Array de features escaladas
    """
    return scaler.transform(X)


def save_scaler(scaler: StandardScaler, filepath: str) -> None:
    """Salva um scaler em arquivo.

    Args:
        scaler: StandardScaler fitado
        filepath: Caminho para salvar o arquivo
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: str) -> StandardScaler:
    """Carrega um scaler de arquivo.

    Args:
        filepath: Caminho para o arquivo do scaler

    Returns:
        StandardScaler carregado
    """
    return joblib.load(filepath)
