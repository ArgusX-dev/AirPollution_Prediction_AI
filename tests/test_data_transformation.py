import numpy as np
from unittest.mock import MagicMock, patch
from air_quality.components.data_transformation import DataTransformation


def test_feature_engineering_time_series(dummy_air_quality_data, mock_schema):
    mock_artifact = MagicMock()
    mock_config = MagicMock()

    with patch('air_quality.components.data_transformation.read_yaml_file', return_value=mock_schema), \
            patch('air_quality.components.data_transformation.TARGET_COLUMN', 'risk_severity'), \
            patch('air_quality.components.data_transformation.DATA_TRANSFORMATION_TARGET_LAG', 24):
        transformation = DataTransformation(mock_artifact, mock_config)
        transformed_df = transformation._feature_engineering_time_series(dummy_air_quality_data)


        assert 'hour' in transformed_df.columns
        assert 'day_of_week' in transformed_df.columns
        assert 'month' in transformed_df.columns
        assert 'risk_severity_lag_24h' in transformed_df.columns

        assert len(transformed_df) == len(dummy_air_quality_data) - 24

        for col in mock_schema['drop_columns']:
            assert col not in transformed_df.columns