"""Arquitetura Perceptron Multicamadas (MLP) para classificação binária.

Este módulo define a arquitetura de rede neural para predição de churn.
Duas classes são fornecidas:
- MLP: Modelo base para inferência (sigmoid na saída)
- MLPForTraining: Wrapper de treino com BCEWithLogitsLoss

A arquitetura segue um padrão padrão:
Linear -> (BatchNorm) -> ReLU -> Dropout -> ... -> Linear -> (Sigmoid)

Decisões principais de design:
- BCEWithLogitsLoss combina sigmoid + BCE para estabilidade numérica
- BatchNorm estabiliza o treino e permite learning rates maiores
- Dropout previne overfitting em redes profundas
- Separar modelos de treino e inferência mantém o código limpo
"""

from __future__ import annotations

import torch
from torch import nn

from src.configs.config import MLPConfig


class MLP(nn.Module):
    """Perceptron Multicamadas para inferência de classificação binária.

    Esta é a arquitetura base que aplica ativação sigmoid na camada de saída,
    produzindo diretamente probabilidades no intervalo [0, 1]. Use esta classe
    para inferência/predição após o treino.

    Arquitetura para cada camada oculta:
        Linear(in_features, out_features)
        -> BatchNorm1d(out_features)  [opcional]
        -> ReLU()
        -> Dropout(p)

    Camada de saída:
        Linear(hidden_dim, 1)
        -> Sigmoid()

    Attributes (Atributos):
        config: MLPConfig contendo parâmetros da arquitetura.
        hidden_layers: Módulo Sequential contendo todas as camadas ocultas.
        output_layer: Camada Linear produzindo logits (1 unidade de saída).
        sigmoid: Ativação Sigmoid para saída de probabilidade.

    Note (Nota):
        Para treino, use MLPForTraining em vez disso para obter os benefícios
        de estabilidade numérica do BCEWithLogitsLoss.

    Exemplo:
        >>> config = MLPConfig(input_dim=45, hidden_dims=(128, 64))
        >>> model = MLP(config)
        >>> model.eval()
        >>> probabilities = model(features)
    """

    def __init__(self, config: MLPConfig) -> None:
        """Inicializa a arquitetura MLP.

        Args:
            config: Objeto de configuração contendo:
                - input_dim: Número de features de entrada
                - hidden_dims: Tupla de tamanhos das camadas ocultas
                - dropout_rate: Probabilidade de dropout
                - use_batch_norm: Se deve usar normalização de batch
        """
        super().__init__()
        self.config = config

        # Constrói dinamicamente as camadas ocultas baseado na configuração
        # Cada bloco oculto: Linear -> (BatchNorm) -> ReLU -> Dropout
        layers: list[nn.Module] = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            # Transformação linear: projeta entrada para dimensão oculta
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # BatchNorm: normaliza ativações entre camadas
            # Benefícios: convergência mais rápida, reduz
            # sensibilidade à inicialização, leve efeito de
            # regularização via ruído nas estatísticas de batch
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # ReLU: ativação não-linear, escolha padrão para camadas ocultas
            # Ativações esparsas ajudam a prevenir gradientes que desaparecem
            layers.append(nn.ReLU())

            # Dropout: zera aleatoriamente ativações durante o treino
            # Previne overfitting quebrando co-adaptação de neurônios
            # Valores típicos: 0.2-0.5 para camadas ocultas
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Camada de saída: unidade única para classificação binária
        # Usa apenas Linear, sem BatchNorm/Dropout na saída
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem forward retornando probabilidades de classe.

        Args:
            x: Tensor de features de entrada de shape (batch_size, input_dim).

        Returns:
            Predições de probabilidade de shape (batch_size,),
            valores em [0, 1].
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return self.sigmoid(logits)

    def get_num_parameters(self) -> int:
        """Conta o total de parâmetros treináveis no modelo.

        Útil para comparar tamanhos de modelos e verificar
        super-parametrização. Regra geral: comece com < 10x
        amostras de treino em parâmetros.

        Returns:
            Número total de parâmetros em todas as camadas.
        """
        return sum(p.numel() for p in self.parameters())


class MLPForTraining(nn.Module):
    """Wrapper de treino para MLP usando BCEWithLogitsLoss.

    Este wrapper separa a lógica de treino do modelo de inferência. Usa
    BCEWithLogitsLoss que é numericamente mais estável que sigmoid + BCE
    separados, especialmente para valores de predição extremos.

    A vantagem do BCEWithLogitsLoss:
        Em vez de: loss = BCE(sigmoid(logits), targets)
        Usamos:    loss = BCEWithLogits(logits, targets)

    Isso evita computar log(sigmoid(x)) que pode ser numéricamente instável
    para logits muito positivos ou muito negativos.

    Attributes (Atributos):
        config: MLPConfig contendo parâmetros da arquitetura.
        model: O modelo MLP subjacente.
        criterion: Função de perda BCEWithLogitsLoss.

    Exemplo:
        >>> config = MLPConfig(input_dim=45, hidden_dims=(128, 64))
        >>> model = MLPForTraining(config)
        >>> outputs = model(features, targets=labels)
        >>> loss = outputs['loss']
        >>> probs = outputs['probs']
    """

    def __init__(self, config: MLPConfig) -> None:
        """Inicializa o wrapper de treino.

        Args:
            config: Configuração para a arquitetura MLP subjacente.
        """
        super().__init__()
        self.config = config
        self.model = MLP(config)

        # BCEWithLogitsLoss: combina sigmoid + entropia cruzada binária
        # Mais numericamente estável que sigmoid manual + BCE
        # Aplica sigmoid internamente durante o cálculo da perda
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Passagem forward: logits, probs e opcionalmente perda.

        Args:
            x: Tensor de features de entrada de shape (batch_size,
                input_dim).
            targets: Rótulos verdadeiros opcionais de shape
                (batch_size,). Se fornecidos, a perda é computada.

        Returns:
            Dicionário contendo:
                - logits: Saída bruta antes do sigmoid,
                    shape (batch_size,)
                - probs: Predições de probabilidade após sigmoid,
                    shape (batch_size,)
                - loss: Tensor escalar de perda (apenas se targets)
        """
        # Passa pelas camadas ocultas (compartilhado com modelo de inferência)
        hidden = self.model.hidden_layers(x)

        # Obtém logits brutos da camada de saída (sem sigmoid ainda)
        # Shape: (batch_size,) após squeeze remover dimensão 1
        logits = self.model.output_layer(hidden).squeeze(-1)

        # Aplica sigmoid para obter probabilidades para predições
        probs = torch.sigmoid(logits)

        # Empacota resultados em dicionário para flexibilidade
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": probs,
        }

        # Computa perda se tivermos rótulos (modo treino)
        if targets is not None:
            # BCEWithLogitsLoss espera targets float e shape compatível
            loss = self.criterion(logits, targets.float())
            result["loss"] = loss

        return result
