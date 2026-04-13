import sys, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.constant.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH, DATA_TRANSFORMATION_TARGET_LAG
from air_quality.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from air_quality.entity.config_entity import DataTransformationConfig
from air_quality.utils.main_utils.utils import save_numpy_array_data, save_object, read_yaml_file


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise AirQualityException(e, sys)

    @staticmethod
    def read_csv(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AirQualityException(e, sys)

    def _feature_engineering_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df['date_hour'] = pd.to_datetime(df['date_hour'])
            df.set_index('date_hour', inplace=True)

            df = df.resample('h').asfreq()
            df.interpolate(method='linear', limit=3, inplace=True)

            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

            target = TARGET_COLUMN
            df[f'{target}_lag_1h'] = df[target].shift(1)
            df[f'{target}_lag_2h'] = df[target].shift(2)
            df[f'{target}_lag_{DATA_TRANSFORMATION_TARGET_LAG}h'] = df[target].shift(DATA_TRANSFORMATION_TARGET_LAG)

            df.dropna(subset=[f'{target}_lag_{DATA_TRANSFORMATION_TARGET_LAG}h'], inplace=True)

            df.dropna(subset=[target], inplace=True)

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=[target], inplace=True)

            columns_to_drop = [col for col in self.schema.get('drop_columns', []) if col in df.columns]
            df.drop(columns=columns_to_drop, inplace=True)

            return df.reset_index(drop=True)

        except Exception as e:
            raise AirQualityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logger.info('Initializing object transform')
            scaler = RobustScaler()
            preprocessor = Pipeline(steps=[
                ('scaler', scaler),
            ])
            return preprocessor
        except Exception as e:
            raise AirQualityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=== STARTING DATA TRANSFORMATION ===")
            train_df = self.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_csv(self.data_validation_artifact.valid_test_file_path)

            logger.info("Applying Temporal Feature Engineering to the Train Set")
            train_transformed = self._feature_engineering_time_series(train_df)


            logger.info("Applying Temporal Feature Engineering to the Test Set")
            last_24h_train = train_df.tail(DATA_TRANSFORMATION_TARGET_LAG)
            test_combined = pd.concat([last_24h_train, test_df], axis=0)
            test_transformed = self._feature_engineering_time_series(test_combined)

            input_feature_train_df = train_transformed.drop(TARGET_COLUMN, axis=1)
            target_feature_train_df = train_transformed[TARGET_COLUMN]

            input_feature_test_df = test_transformed.drop(TARGET_COLUMN, axis=1)
            target_feature_test_df = test_transformed[TARGET_COLUMN]

            preprocessor = self.get_data_transformer_object()
            logger.info("Training the preprocessor (RobustScaler)...")

            transform_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transform_input_test_feature = preprocessor.transform(input_feature_test_df)

            train_array = np.c_[transform_input_train_feature, np.array(target_feature_train_df)]
            test_array = np.c_[transform_input_test_feature, np.array(target_feature_test_df)]

            logger.info(
                f"Saving numpy Array... Train Shape: {train_array.shape}, Test Shape: {test_array.shape}")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_array)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            save_object('final_model/preprocessor.pkl', preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logger.info("=== DATA TRANSFORMATION COMPLETED ===")
            return data_transformation_artifact

        except Exception as e:
            raise AirQualityException(e, sys)