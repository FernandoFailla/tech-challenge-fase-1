"""Utilitários de checkpoint para salvar e carregar estado de treinamento.

Este módulo fornece funções para persistir e restaurar estado de treino:
1. Checkpoints completos: Salva modelo + otimizador + época para retomar treino
2. Artefatos do melhor modelo: Salva apenas pesos do modelo para
   inferência/deploy

A distinção é importante:
- Checkpoints capturam o estado COMPLETO de treinamento, permitindo
  pausar e retomar o treino de exatamente onde parou
  (estado do otimizador, learning rate, etc.)
- Salvamento do melhor modelo é mais leve e destinado a deploy -
  contém apenas pesos do modelo e metadados opcionais.

Casos de uso:
- save_checkpoint(): Durante treino para tolerância a falhas
- save_best_model(): Após treino para deploy/inferência
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    filepath: str | Path,
) -> None:
    """Salva estado completo de treinamento para retomar posteriormente.

    Salva todos os componentes necessários para retomar treino: pesos do
    modelo,
    estado do otimizador (que inclui buffers de momentum para Adam),
    época atual e melhor score de validação. Essencial para:
    - Tolerância a falhas (recuperação de crash)
    - Treinos longos que precisam pausar
    - Retomar treino após ajustar hiperparâmetros

    Args:
        model: Rede neural para salvar pesos.
        optimizer: Otimizador para salvar estado (inclui
            momentum, learning rate).
        epoch: Número da época atual (0-indexado). Usado para
            retomar na posição correta.
        best_score: Melhor score de validação alcançado até agora.
            Usado para restaurar estado do early stopping corretamente.
        filepath: Caminho para salvar arquivo de checkpoint. Diretórios
            pai criados automaticamente.

    Note:
        - Arquivos são relativamente grandes (incluem estado do otimizador)
        - Use save_best_model() para deploy final (menor)
        - Carregue com load_checkpoint() para retomar treino

    Example:
        >>> save_checkpoint(model, optimizer, epoch=10, best_score=0.85,
        ...                 filepath="checkpoints/checkpoint_epoch10.pt")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Salva todo o estado necessário para retomar treino exatamente onde parou
    # model_state_dict: pesos aprendidos
    # optimizer_state_dict: buffers de momentum, estado do learning rate
    # epoch: para retomar da posição correta
    # best_score: para restaurar threshold do early stopping
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
        },
        filepath,
    )


def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Carrega estado de treinamento de arquivo checkpoint.

    Restaura pesos do modelo e opcionalmente estado do otimizador de um
    checkpoint salvo anteriormente. Retorna o dicionário do checkpoint para
    acesso a metadados como época e best_score para restauração adequada do
    estado.

    Args:
        filepath: Caminho para arquivo checkpoint criado por
            save_checkpoint().
        model: Instância do modelo para carregar pesos. Arquitetura deve
            corresponder ao modelo que foi salvo.
        optimizer: Otimizador opcional para carregar estado. Forneça quando
            retomando treino. Omita quando apenas avaliando um checkpoint.
            Se None, estado do otimizador é ignorado.

    Returns:
        Dicionário contendo metadados do checkpoint:
            - epoch: O número da época quando o checkpoint foi salvo
            - best_score: Melhor score de validação no momento do checkpoint
            - model_state_dict: Pesos do modelo (carregados no model)
            - optimizer_state_dict: Estado do otimizador (se fornecido)

    Raises:
        FileNotFoundError: Se arquivo checkpoint não existir.
        RuntimeError: Se arquitetura do modelo não corresponder aos pesos
            salvos.

    Example:
        >>> checkpoint = load_checkpoint("model.pt", model, optimizer=None)
        >>> print(f"Retomando da época {checkpoint['epoch']}")
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Estado do otimizador só necessário para retomar treino, não para
    # inferência
    # Contém buffers de momentum para Adam, learning rates por parâmetro, etc.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


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
