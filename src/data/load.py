"""Carregamento do dataset Telco Customer Churn."""

from __future__ import annotations

import pandas as pd


def load_telco_data(
    filepath: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    """Carrega o dataset Telco Customer Churn de arquivo CSV.

    Args:
        filepath: Caminho para o arquivo CSV.
            Default: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

    Returns:
        DataFrame pandas com o dataset carregado

    Raises:
        FileNotFoundError: Se o arquivo não existir. Inclui
            mensagem sugerindo rodar 'dvc pull'.

    Example:
        >>> df = load_telco_data()
        >>> print(df.shape)
        (7043, 21)
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Arquivo '{filepath}' não encontrado. "
            "Você lembrou de rodar 'dvc pull' para baixar os dados localmente?"
        ) from e


if __name__ == "__main__":  # pragma: no cover
    df = load_telco_data()
    print(df.head())
