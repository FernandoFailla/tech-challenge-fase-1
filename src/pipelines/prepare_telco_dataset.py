from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_DVC_PATH = REPO_ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc"
EXPECTED_SHA256 = (
    "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
)


def calculate_sha256(file_path: Path) -> str:
    """Calcula o SHA256 de um arquivo."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def validate_dataset(path: Path = DATASET_PATH) -> str:
    """Valida existência e integridade do dataset via SHA256."""
    if not path.exists():
        if DATASET_DVC_PATH.exists():
            msg = (
                f"Dataset não encontrado em: {path}. "
                "Este projeto usa DVC para esse arquivo; execute `dvc pull`."
            )
            raise FileNotFoundError(msg)

        msg = f"Dataset não encontrado em: {path}"
        raise FileNotFoundError(msg)

    checksum = calculate_sha256(path)
    if checksum != EXPECTED_SHA256:
        msg = (
            "Checksum inválido. "
            f"Esperado: {EXPECTED_SHA256} | Obtido: {checksum}"
        )
        raise ValueError(msg)

    return checksum


def main() -> None:
    """Executa validação simples do dataset e imprime checksum."""
    checksum = validate_dataset()
    print(f"Dataset validado em: {DATASET_PATH}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
