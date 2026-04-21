"""Utilitários de checkpoint para salvar e carregar estado de treinamento."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_best_model(
    model: torch.nn.Module,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Salva modelo para deploy/inferência (sem estado do otimizador).

    Cria artefato leve contendo apenas pesos do modelo e metadados opcionais.
    Este é o artefato final a ser usado para:
    - Servir modelo/deploy
    - Scripts de inferência
    - Registro de modelo (MLflow, etc.)

    Args:
        model: Modelo treinado para salvar pesos.
        filepath: Caminho de destino. Diretórios pai criados automaticamente.
        metadata: Dicionário opcional de metadados de treino para salvar junto
            aos pesos do modelo. Útil para armazenar:
            - epoch: Quando modelo alcançou melhor performance
            - metrics: Melhor loss de validação, F1, etc.
            - config: Hiperparâmetros de treino usados
            - feature info: Nomes de features, parâmetros de pré-processamento

    Note:
        - Muito menor que checkpoints (sem estado do otimizador)
        - Não pode retomar treino destes arquivos (use checkpoints para isso)
        - Formato padrão para artefatos de deploy

    Example:
        >>> save_best_model(
        ...     model,
        ...     "models/production_model.pt",
        ...     metadata={"epoch": 42, "val_loss": 0.234, "val_f1": 0.89}
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Salva estado mínimo para deploy de inferência
    # Sem estado do otimizador torna arquivos menores e carregamento mais
    # rápido
    save_dict: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    # Metadados ajudam a rastrear versão e performance do modelo
    # Não necessários para inferência, mas úteis para gestão de modelo
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, filepath)
