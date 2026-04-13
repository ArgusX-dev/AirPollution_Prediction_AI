import os,sys
import numpy as np
import pandas as pd

'''
Common constants variables
'''

TARGET_COLUMN: str = 'risk_severity'
PIPELINE_NAME: str = 'Air_Quality_Risk_Prediction' # Actualizado
ARTIFACT_DIR: str = 'artifacts'
FILE_NAME: str = 'weather_data.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
SCHEMA_FILE_PATH = os.path.join('data_schema','schema.yaml')


'''
Data Ingestion related constants
'''
DATA_INGESTION_TABLE_NAME: str = 'weather_pollution'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

'''
Data Validation Related constants
'''
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR: str = 'validated'
DATA_VALIDATION_INVALID_DIR: str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR: str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = 'report_data.yaml'

'''
Data Transformation related constants
'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
DATA_TRANSFORMATION_TARGET_LAG: int = 24
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


'''
Model Trainer Related constants
'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
MODEL_FILE_NAME: str = "model.pkl"
TRAINING_BUCKET_NAME = "netwworksecurity"
SAVED_MODEL_DIR =os.path.join("saved_models")

'''
Model Pusher Related constants
'''
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
BUCKET_NAME: str = "airpollutionpredictor"
S3_MODEL_DIR_NAME: str = "final_model"