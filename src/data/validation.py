"""Validações genéricas de dados tabulares."""

from __future__ import annotations

import pandas as pd


def validate_required_columns(df: pd.DataFrame, target_column: str) -> None:
    """Valida se o dataframe possui dados e coluna alvo."""
    if df.empty:
        raise ValueError("Dataset vazio. Não é possível treinar.")

    if target_column not in df.columns:
        msg = f"Coluna alvo ausente no dataset: '{target_column}'."
        raise ValueError(msg)


def validate_binary_target(
    y: pd.Series,
    positive_label: str | int,
    min_target_classes: int = 2,
) -> None:
    """Valida presença da classe positiva e cardinalidade mínima do alvo."""
    unique_values = set(y.astype(str).unique().tolist())
    if str(positive_label) not in unique_values:
        msg = (
            f"Classe positiva '{positive_label}' não encontrada "
            "na coluna alvo. "
            "Verifique o mapeamento do target."
        )
        raise ValueError(msg)

    if len(unique_values) < min_target_classes:
        raise ValueError("A coluna alvo precisa ter pelo menos duas classes.")
