from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.training.mlp.checkpoint import save_best_model


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.fast
def test_save_best_model(tmp_path: Path) -> None:
    model = SimpleModel()
    model_path = tmp_path / "best_model.pt"
    accuracy = 0.95

    save_best_model(model, model_path, metadata={"accuracy": accuracy})

    assert model_path.exists()

    loaded_data = torch.load(model_path, weights_only=False)
    assert "model_state_dict" in loaded_data
    assert loaded_data["metadata"]["accuracy"] == accuracy
