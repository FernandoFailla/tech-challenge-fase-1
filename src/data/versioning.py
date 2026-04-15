"""Funções genéricas de versionamento de dataset."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_DVC_METADATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc"


def get_dataset_version_from_dvc(
    dvc_path: str = DEFAULT_DVC_METADATA_PATH,
) -> str:
    """Lê versão do dataset a partir do arquivo .dvc (md5)."""
    with Path(dvc_path).open(encoding="utf-8") as dvc_file:
        dvc_metadata = yaml.safe_load(dvc_file)

    outs = dvc_metadata.get("outs", [])
    if not outs:
        raise ValueError("Arquivo .dvc inválido: seção 'outs' ausente.")

    md5_value = outs[0].get("md5")
    if not md5_value:
        raise ValueError("Arquivo .dvc inválido: campo 'md5' ausente.")

    return str(md5_value)
