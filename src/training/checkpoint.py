"""Model checkpointing utilities for saving and loading training state.

This module provides functions to persist and restore model training state:
1. Full checkpoints: Save model + optimizer + epoch for resuming training
2. Best model artifacts: Save only model weights for inference/deployment

The distinction is important:
- Checkpoints capture the FULL training state, allowing you to
  pause and resume training from exactly where you left off
  (optimizer state, learning rate, etc.)
- Best model saves are lighter and meant for deployment -
  they contain only the model weights and optional metadata.

Use cases:
- save_checkpoint(): During training for fault tolerance (resume after crash)
- save_best_model(): After training for deployment/inference
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
    """Save complete training state for resuming training later.

    Saves all components needed to resume training: model weights,
    optimizer state (which includes momentum buffers for Adam),
    current epoch, and best validation score. This is essential for:
    - Fault tolerance (crash recovery)
    - Long training jobs that need pausing
    - Resuming training after adjusting hyperparameters

    Args:
        model: Neural network to save weights for.
        optimizer: Optimizer to save state from (includes
            momentum, learning rate).
        epoch: Current epoch number (0-indexed). Used to
            resume at correct position.
        best_score: Best validation score achieved so far.
            Used to restore early stopping state correctly.
        filepath: Path to save checkpoint file. Parent
            directories created automatically.

    Note:
        - Files are relatively large (includes optimizer state)
        - Use save_best_model() for final deployment (smaller)
        - Load with load_checkpoint() to resume training

    Example:
        >>> save_checkpoint(model, optimizer, epoch=10, best_score=0.85,
        ...                 filepath="checkpoints/checkpoint_epoch10.pt")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save all state needed to resume training exactly where we left off
    # model_state_dict: learned weights
    # optimizer_state_dict: momentum buffers, learning rate state
    # epoch: to resume from correct position
    # best_score: to restore early stopping threshold
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
    """Load training state from checkpoint file.

    Restores model weights and optionally optimizer state from a previously
    saved checkpoint. Returns the checkpoint dictionary for access to metadata
    like epoch and best_score for proper state restoration.

    Args:
        filepath: Path to checkpoint file created by save_checkpoint().
        model: Model instance to load weights into. Architecture must match
            the model that was saved.
        optimizer: Optional optimizer to load state into. Provide this when
            resuming training. Omit when only evaluating a checkpoint.
            If None, optimizer state is ignored.

    Returns:
        Dictionary containing checkpoint metadata:
            - epoch: The epoch number when checkpoint was saved
            - best_score: Best validation score at checkpoint time
            - model_state_dict: Model weights (loaded into model)
            - optimizer_state_dict: Optimizer state (if provided)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If model architecture doesn't match saved weights.

    Example:
        >>> checkpoint = load_checkpoint("model.pt", model, optimizer=None)
        >>> print(f"Resuming from epoch {checkpoint['epoch']}")
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Optimizer state only needed for resuming training, not for inference
    # Contains momentum buffers for Adam, per-parameter learning rates, etc.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_best_model(
    model: torch.nn.Module,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model for deployment/inference (without optimizer state).

    Creates a lightweight artifact containing only model weights and optional
    metadata. This is the final artifact to use for:
    - Model serving/deployment
    - Inference scripts
    - Model registry (MLflow, etc.)

    Args:
        model: Trained model to save weights from.
        filepath: Destination path. Parent directories created automatically.
        metadata: Optional dictionary of training metadata to save alongside
            model weights. Useful for storing:
            - epoch: When model achieved best performance
            - metrics: Best validation loss, F1, etc.
            - config: Training hyperparameters used
            - feature info: Feature names, preprocessing parameters

    Note:
        - Much smaller than checkpoints (no optimizer state)
        - Cannot resume training from these files (use checkpoints for that)
        - Standard format for model deployment artifacts

    Example:
        >>> save_best_model(
        ...     model,
        ...     "models/production_model.pt",
        ...     metadata={"epoch": 42, "val_loss": 0.234, "val_f1": 0.89}
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save minimal state for inference deployment
    # No optimizer state makes files smaller and loading faster
    save_dict: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    # Metadata helps track model version and performance
    # Not needed for inference, but useful for model management
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, filepath)
