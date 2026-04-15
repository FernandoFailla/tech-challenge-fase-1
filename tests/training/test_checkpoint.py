from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.training.checkpoint import (
    load_checkpoint,
    save_best_model,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.fast
def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = tmp_path / "checkpoint.pt"
    epoch = 5
    best_score = 0.95

    save_checkpoint(
        model,
        optimizer,
        epoch=epoch,
        best_score=best_score,
        filepath=checkpoint_path,
    )

    assert checkpoint_path.exists()

    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

    checkpoint = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert checkpoint["epoch"] == epoch
    assert checkpoint["best_score"] == best_score

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


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
