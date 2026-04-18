"""MLP (Multi-Layer Perceptron) training module.

Este pacote contém tudo relacionado ao treinamento de modelos MLP:
- model: Definição da arquitetura MLP (MLP, MLPForTraining)
- trainer: Treinador com loop de treino, early stopping, etc.
- checkpoint: Salvamento de modelos PyTorch
- early_stopping: Callback de parada antecipada
"""

from __future__ import annotations

from src.training.mlp.checkpoint import save_best_model
from src.training.mlp.early_stopping import EarlyStopping
from src.training.mlp.model import MLP, MLPForTraining
from src.training.mlp.trainer import MLPTrainer

__all__ = [
    "MLP",
    "EarlyStopping",
    "MLPForTraining",
    "MLPTrainer",
    "save_best_model",
]
