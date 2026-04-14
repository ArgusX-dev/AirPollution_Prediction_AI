import os
import pandas as pd
from unittest.mock import patch, MagicMock
from air_quality.components.data_ingestion import DataIngestion



class MockDataIngestionConfig:
    table_name = "weather_pollution"
    feature_store_file_path = "artifacts/data_ingestion/feature_store/weather_data.csv"
    training_file_path = "artifacts/data_ingestion/ingested/train.csv"
    testing_file_path = "artifacts/data_ingestion/ingested/tests.csv"
    train_test_split_ratio = 0.2


@patch('air_quality.components.data_ingestion.get_db_engine')
@patch('air_quality.components.data_ingestion.pd.read_sql')
def test_data_ingestion_pipeline(mock_read_sql, mock_get_engine, dummy_air_quality_data, tmp_path):
    mock_read_sql.return_value = dummy_air_quality_data
    mock_get_engine.return_value = MagicMock()

    config = MockDataIngestionConfig()
    config.feature_store_file_path = os.path.join(tmp_path, "feature_store.csv")
    config.training_file_path = os.path.join(tmp_path, "train.csv")
    config.testing_file_path = os.path.join(tmp_path, "tests.csv")

    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()

    assert mock_read_sql.called
    assert os.path.exists(config.feature_store_file_path)
    assert os.path.exists(config.training_file_path)
    assert os.path.exists(config.testing_file_path)

    df_train = pd.read_csv(config.training_file_path)
    df_test = pd.read_csv(config.testing_file_path)

    assert len(df_train) == 80
    assert len(df_test) == 20
    assert artifact.train_file_path == config.training_file_path