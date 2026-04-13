from air_quality.components.data_ingestion import DataIngestion
from air_quality.components.data_validation import DataValidation
from air_quality.components.data_transformation import DataTransformation
from air_quality.components.model_pusher import ModelPusher
from air_quality.components.model_trainer import ModelTrainer
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
import os,sys
from air_quality.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelPusherConfig,ModelTrainerConfig
from air_quality.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()


    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info('DataIngestion Initialized')
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info('Data Ingestion Completed Successfully')
            logger.info(f'data_ingestion_artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise AirQualityException(e,sys)

    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info('DataValidation Initialized')
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info('Data Validation Completed Successfully')
            logger.info(f'data_validation_config: {data_validation_config}')
            return data_validation_artifact
        except Exception as e:
            raise AirQualityException(e,sys)
    def start_data_transformation(self,data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info('DataTransformation Initialized')
            data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact, data_transformation_config = data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info('Data Transformation Completed Successfully')
            logger.info(f'data_transformation: {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as e:
            raise AirQualityException(e,sys)

    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact)-> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info('ModelTrainerConfig Initialized')
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info('Model Trainer Completed Successfully')
            logger.info(f'model_trainer_artifact: {model_trainer_artifact}')
            return model_trainer_artifact
        except Exception as e:
            raise AirQualityException(e,sys)

    def start_model_pusher(self,model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info('ModelPusherConfig Initialized')
            model_pusher = ModelPusher(model_trainer_artifact=model_trainer_artifact, model_pusher_config=model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info('Model Pusher Completed Successfully')
            logger.info(f'model_pusher_artifact: {model_pusher_artifact}')
            return model_pusher_artifact
        except Exception as e:
            raise AirQualityException(e,sys)

    def run_pipeline(self):
        try:
            logger.info("=== STARTING TRAINING PIPELINE ===")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_pusher_artifact = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
            logger.info("=== PIPELINE COMPLETED ===")
            return model_trainer_artifact, model_pusher_artifact
        except Exception as e:
            raise AirQualityException(e,sys)