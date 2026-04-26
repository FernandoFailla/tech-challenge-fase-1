from __future__ import annotations

import pytest

from src.training.mlp.early_stopping import EarlyStopping


@pytest.mark.fast
def test_should_not_stop_early() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")

    assert not es(1.0)
    assert not es(0.9)
    assert not es(0.8)
    assert not es(0.7)
    assert not es.early_stop


@pytest.mark.fast
def test_should_stop_after_patience() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")

    es(1.0)
    es(1.0)
    es(1.0)
    es(1.0)
    assert es.early_stop


@pytest.mark.fast
def test_is_best_updates() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")

    assert es.is_best(1.0)
    _ = es(1.0)
    assert not es.is_best(1.1)
    assert es.is_best(0.9)
    _ = es(0.9)

    expected_best = 0.9
    assert es.best_score == expected_best
    assert es.counter == 0


@pytest.mark.fast
def test_reset() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")

    es(1.0)
    es(1.0)
    es(1.0)
    es(1.0)

    assert es.early_stop

    es.reset()

    assert es.counter == 0
    assert es.best_score is None
    assert not es.early_stop


@pytest.mark.fast
def test_max_mode() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0, mode="max")

    assert not es(0.5)
    assert not es(0.6)
    assert not es(0.7)
    es(0.6)
    es(0.6)
    es(0.6)

    assert es.early_stop
