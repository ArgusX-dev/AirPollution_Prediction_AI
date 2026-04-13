import os,sys
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from air_quality.cloud.database import get_db_engine
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.entity.config_entity import DataIngestionConfig
from air_quality.entity.artifact_entity import DataIngestionArtifact





load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.engine = get_db_engine()
        except Exception as e:
            raise AirQualityException(e, sys)

    def export_table_as_dataframe(self) -> pd.DataFrame:
        try:
            logger.info("Starting data extraction from AWS RDS...")
            table_name = self.data_ingestion_config.table_name

            query = f"SELECT * FROM {table_name} ORDER BY date_hour ASC"
            df = pd.read_sql(query, self.engine)

            logger.info(f"Successful extraction. DataFrame shape: {df.shape}")
            return df

        except Exception as e:
            raise AirQualityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Raw data stored in Feature Store: {feature_store_file_path}")

            return dataframe
        except Exception as e:
            raise AirQualityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logger.info("Starting Train-Test Split of the dataframe...")
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,shuffle=False
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logger.info("Train and Test sets successfully exported.")

        except Exception as e:
            raise AirQualityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("=== STARTING DATA INGESTION COMPONENT ===")

            dataframe = self.export_table_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logger.info(f"Ingestion artifact created: {data_ingestion_artifact}")
            logger.info("=== COMPLETED INGESTION COMPONENT ===")

            return data_ingestion_artifact

        except Exception as e:
            raise AirQualityException(e, sys)