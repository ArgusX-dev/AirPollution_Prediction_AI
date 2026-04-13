import os,sys
import boto3
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from air_quality.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_pusher_config: ModelPusherConfig):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_pusher_config = model_pusher_config
            self.s3_client = boto3.client('s3')
            sts_client = boto3.client('sts')
            identidad = sts_client.get_caller_identity()
            logger.info(f"EJECUTANDO AWS COMO: {identidad['Arn']}")
        except Exception as e:
            raise AirQualityException(e, sys)

    def _upload_to_s3(self, local_file_path: str, s3_key: str):
        try:
            bucket = self.model_pusher_config.bucket_name
            logger.info(f"Uploading {local_file_path} a s3://{bucket}/{s3_key} ...")

            self.s3_client.upload_file(local_file_path, bucket, s3_key)

            logger.info(f"Success: {s3_key}")
        except Exception as e:
            raise AirQualityException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("=== INITIALIZING MODEL PUSHER ===")

            local_model_path = 'final_model/model.pkl'
            local_preprocessor_path = 'final_model/preprocessor.pkl'

            if not os.path.exists(local_model_path) or not os.path.exists(local_preprocessor_path):
                raise Exception("Files were not found in the final_model/ folder")

            s3_model_path = f"{self.model_pusher_config.s3_model_key_path}/model.pkl"
            s3_preprocessor_path = f"{self.model_pusher_config.s3_model_key_path}/preprocessor.pkl"

            self._upload_to_s3(local_model_path, s3_model_path)
            self._upload_to_s3(local_preprocessor_path, s3_preprocessor_path)

            model_pusher_artifact = ModelPusherArtifact(
                s3_model_path=f"s3://{self.model_pusher_config.bucket_name}/{self.model_pusher_config.s3_model_key_path}",
                model_pushed=True
            )

            logger.info(f"Artifact Model Pusher: {model_pusher_artifact}")
            logger.info("=== MODEL PUSHER COMPLETED ===")

            return model_pusher_artifact

        except Exception as e:
            raise AirQualityException(e, sys)