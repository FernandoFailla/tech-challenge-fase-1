from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


def calculate_sha256(file_path: Path) -> str:
    """Calcula o SHA256 de um arquivo."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def prepare_dataset(source: Path, destination: Path) -> str:
    """Copia dataset para data/raw e retorna checksum SHA256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return calculate_sha256(destination)


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Copia e valida o dataset Telco Customer Churn.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Caminho de origem do CSV bruto.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Caminho de destino no repositório.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o preparo e imprime checksum final."""
    args = parse_args()
    checksum = prepare_dataset(args.source, args.destination)
    print(f"Dataset preparado em: {args.destination}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
