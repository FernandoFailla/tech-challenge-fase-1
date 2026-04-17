"""Funções genéricas para split de dados."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test_stratified(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste com seed fixa e estratificação no alvo."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
