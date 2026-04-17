"""Tests for data validation module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import (
    validate_binary_target,
    validate_required_columns,
)


def test_validate_required_columns_empty_df() -> None:
    """Should raise ValueError for empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset vazio"):
        validate_required_columns(df, "Churn")


def test_validate_required_columns_missing_target() -> None:
    """Should raise ValueError for missing target column."""
    df = pd.DataFrame({"gender": ["Female", "Male"]})
    with pytest.raises(ValueError, match="Coluna alvo ausente"):
        validate_required_columns(df, "Churn")


def test_validate_required_columns_valid() -> None:
    """Should not raise for valid DataFrame with target."""
    df = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "Churn": ["No", "Yes"],
        }
    )
    # Should not raise
    validate_required_columns(df, "Churn")


def test_validate_binary_target_missing_positive() -> None:
    """Should raise ValueError when positive label not found."""
    y = pd.Series(["No", "No", "No"])
    with pytest.raises(ValueError, match="Classe positiva"):
        validate_binary_target(y, "Yes")


def test_validate_binary_target_single_class() -> None:
    """Should raise ValueError for single class target."""
    y = pd.Series(["Yes", "Yes", "Yes"])
    with pytest.raises(ValueError, match="pelo menos duas classes"):
        validate_binary_target(y, "Yes")


def test_validate_binary_target_valid() -> None:
    """Should not raise for valid binary target."""
    y = pd.Series(["Yes", "No", "Yes", "No"])
    # Should not raise
    validate_binary_target(y, "Yes")


def test_validate_binary_target_with_numeric_labels() -> None:
    """Should work with numeric labels."""
    y = pd.Series([1, 0, 1, 0])
    validate_binary_target(y, 1)
