import pandas as pd  # type: ignore[import-untyped]


def load_telco_data(
    filepath: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Arquivo '{filepath}' não encontrado. "
            "Você lembrou de rodar 'dvc pull' para baixar os dados localmente?"
        ) from e


if __name__ == "__main__":
    df = load_telco_data()
    print(df.head())
