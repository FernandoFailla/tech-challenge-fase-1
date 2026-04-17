"""Tests for data splitting module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splitting import split_train_test_stratified


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Create minimal DataFrame for split tests."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1],
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


def test_split_data_returns_expected_sizes(dummy_df: pd.DataFrame) -> None:
    """Split should return sets with consistent sizes."""
    target_column = "Churn"
    test_size = 0.5
    random_seed = 42

    X_train, X_test, y_train, y_test = split_train_test_stratified(
        dummy_df,
        target_column,
        test_size,
        random_seed,
    )

    expected_size = 3
    assert len(X_train) == expected_size
    assert len(X_test) == expected_size
    assert len(y_train) == expected_size
    assert len(y_test) == expected_size


def test_split_data_preserves_target_column(dummy_df: pd.DataFrame) -> None:
    """Split should not include target in X DataFrames."""
    target_column = "Churn"

    X_train, X_test, y_train, y_test = split_train_test_stratified(
        dummy_df,
        target_column,
        test_size=0.5,
        random_seed=42,
    )

    assert target_column not in X_train.columns
    assert target_column not in X_test.columns
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_split_data_reproducibility(dummy_df: pd.DataFrame) -> None:
    """Split should be reproducible with same seed."""
    target_column = "Churn"
    test_size = 0.5
    random_seed = 42

    X_train1, X_test1, y_train1, y_test1 = split_train_test_stratified(
        dummy_df,
        target_column,
        test_size,
        random_seed,
    )

    X_train2, X_test2, y_train2, y_test2 = split_train_test_stratified(
        dummy_df,
        target_column,
        test_size,
        random_seed,
    )

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)
