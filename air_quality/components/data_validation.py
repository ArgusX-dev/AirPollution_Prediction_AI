from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from air_quality.entity.config_entity import DataValidationConfig
from air_quality.constant.training_pipeline import SCHEMA_FILE_PATH
import os,sys
import pandas as pd
from scipy.stats import ks_2samp
from air_quality.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise AirQualityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AirQualityException(e, sys)

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_columns = list(self.schema_config['columns'].keys())
            dataframe_columns = list(dataframe.columns)

            missing_columns = [col for col in schema_columns if col not in dataframe_columns]

            if len(missing_columns) > 0:
                logger.error(f"The following required columns are missing: {missing_columns}")
                return False

            logger.info("All required columns are present in the DataFrame.")
            return True
        except Exception as e:
            raise AirQualityException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            exclude_columns = ['id', 'date_hour', 'register_date', 'risk_category', 'main_pollutant']

            for column in base_df.columns:
                if column in exclude_columns:
                    continue

                d1 = base_df[column]
                d2 = current_df[column]

                sample_size = min(1000, len(d1), len(d2))
                d1_sample = d1.sample(n=sample_size, random_state=42)
                d2_sample = d2.sample(n=sample_size, random_state=42)

                from scipy.stats import ks_2samp
                is_same_dist = ks_2samp(d1_sample, d2_sample)

               
                if is_same_dist.pvalue < threshold:
                    is_drift = True
                    status = False
                else:
                    is_drift = False

                report[column] = {
                    'p_value': float(is_same_dist.pvalue),
                    'drift_status': is_drift
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise AirQualityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("=== STARTING DATA VALIDATION ===")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            status_train = self.validate_columns(dataframe=train_dataframe)
            status_test = self.validate_columns(dataframe=test_dataframe)

            validation_status = status_train and status_test

            if validation_status:
                logger.info("Valid data. Moving to the Validated Data folder.")
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

                valid_train_path = self.data_validation_config.valid_train_file_path
                valid_test_path = self.data_validation_config.valid_test_file_path
                invalid_train_path = None
                invalid_test_path = None

            else:
                logger.error("Invalid data (incorrect schema). Moving to Invalid Data.")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False, header=True)

                valid_train_path = None
                valid_test_path = None
                invalid_train_path = self.data_validation_config.invalid_train_file_path
                invalid_test_path = self.data_validation_config.invalid_test_file_path

            drift_status = self.data_drift(base_df=train_dataframe, current_df=test_dataframe)
            logger.info(
                f"Data Drift report generated. General drift status: {'Detected' if not drift_status else 'Not Drift'}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=invalid_train_path,
                invalid_test_file_path=invalid_test_path,
                drift_file_path=self.data_validation_config.drift_report_file_path
            )

            logger.info("=== DATA VALIDATION COMPLETED ===")

            if not validation_status:
                raise Exception(
                    "The pipeline stopped because the data failed schema validation. Check the 'invalid' folder.")

            return data_validation_artifact

        except Exception as e:
            raise AirQualityException(e, sys)



