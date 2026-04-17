"""Tests for data versioning module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.versioning import (
    DEFAULT_DVC_METADATA_PATH,
    get_dataset_version_from_dvc,
)


def test_get_dataset_version_from_dvc_success(tmp_path: Path) -> None:
    """Should extract md5 from valid dvc file."""
    dvc_file = tmp_path / "test.csv.dvc"
    dvc_content = {
        "outs": [
            {
                "md5": "3b0bfab28a8101b4e4fdd08025a5c235",
                "size": 970457,
                "hash": "md5",
                "path": "test.csv",
            }
        ]
    }
    with dvc_file.open("w") as f:
        yaml.dump(dvc_content, f)

    version = get_dataset_version_from_dvc(str(dvc_file))
    assert version == "3b0bfab28a8101b4e4fdd08025a5c235"


def test_get_dataset_version_from_dvc_missing_outs(tmp_path: Path) -> None:
    """Should raise ValueError for missing outs section."""
    dvc_file = tmp_path / "test.csv.dvc"
    dvc_content: dict = {"size": 100}
    with dvc_file.open("w") as f:
        yaml.dump(dvc_content, f)

    with pytest.raises(ValueError, match="seção 'outs' ausente"):
        get_dataset_version_from_dvc(str(dvc_file))


def test_get_dataset_version_from_dvc_missing_md5(tmp_path: Path) -> None:
    """Should raise ValueError for missing md5 field."""
    dvc_file = tmp_path / "test.csv.dvc"
    dvc_content = {"outs": [{"size": 100, "path": "test.csv"}]}
    with dvc_file.open("w") as f:
        yaml.dump(dvc_content, f)

    with pytest.raises(ValueError, match="campo 'md5' ausente"):
        get_dataset_version_from_dvc(str(dvc_file))


def test_get_dataset_version_from_dvc_empty_outs(tmp_path: Path) -> None:
    """Should raise ValueError for empty outs list."""
    dvc_file = tmp_path / "test.csv.dvc"
    dvc_content = {"outs": []}
    with dvc_file.open("w") as f:
        yaml.dump(dvc_content, f)

    with pytest.raises(ValueError, match="seção 'outs' ausente"):
        get_dataset_version_from_dvc(str(dvc_file))


@pytest.mark.skipif(
    not Path(DEFAULT_DVC_METADATA_PATH).exists(),
    reason="DVC file not available",
)
def test_get_dataset_version_from_dvc_real_file() -> None:
    """Should work with actual DVC file if present."""
    version = get_dataset_version_from_dvc()
    MD5_HASH_LENGTH = 32
    assert len(version) == MD5_HASH_LENGTH  # MD5 hash length
    assert all(c in "0123456789abcdef" for c in version)
