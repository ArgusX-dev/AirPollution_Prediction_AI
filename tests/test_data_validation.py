import pytest
import os
from unittest.mock import MagicMock, patch
from air_quality.components.data_validation import DataValidation


def test_validate_columns_success(dummy_air_quality_data, mock_schema):
    mock_artifact = MagicMock()
    mock_config = MagicMock()

    with patch('air_quality.components.data_validation.read_yaml_file', return_value=mock_schema):
        validation = DataValidation(mock_artifact, mock_config)


        assert validation.validate_columns(dummy_air_quality_data) is True


def test_validate_columns_missing(dummy_air_quality_data, mock_schema):
    mock_artifact = MagicMock()
    mock_config = MagicMock()

    with patch('air_quality.components.data_validation.read_yaml_file', return_value=mock_schema):
        validation = DataValidation(mock_artifact, mock_config)

        corrupted_df = dummy_air_quality_data.drop(columns=['temperature_c'])
        assert validation.validate_columns(corrupted_df) is False


def test_data_drift_detection(dummy_air_quality_data, mock_schema, tmp_path):
    mock_artifact = MagicMock()
    mock_config = MagicMock()
    mock_config.drift_report_file_path = os.path.join(tmp_path, "drift.yaml")

    with patch('air_quality.components.data_validation.read_yaml_file', return_value=mock_schema):
        validation = DataValidation(mock_artifact, mock_config)

        status = validation.data_drift(dummy_air_quality_data, dummy_air_quality_data)
        assert status is True
        assert os.path.exists(mock_config.drift_report_file_path)