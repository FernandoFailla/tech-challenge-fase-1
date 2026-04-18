from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.data.load import load_telco_data


def test_load_telco_data_success() -> None:
    """Test successful data loading."""
    mock_df = pd.DataFrame({"customerID": ["1", "2"], "Churn": ["Yes", "No"]})

    with patch(
        "src.data.load.pd.read_csv", return_value=mock_df
    ) as mock_read_csv:
        result = load_telco_data("dummy/path.csv")

        mock_read_csv.assert_called_once_with("dummy/path.csv")
        pd.testing.assert_frame_equal(result, mock_df)


def test_load_telco_data_file_not_found() -> None:
    """Test file not found exception handling."""
    with (
        patch(
            "src.data.load.pd.read_csv",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(FileNotFoundError, match="não encontrado"),
    ):
        load_telco_data("missing_file.csv")
