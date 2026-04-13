import sys

from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.pipeline.training_pipeline import TrainingPipeline

if __name__ == '__main__':
    try:
        logger.info("=== STARTING LOCAL TESTING PIPELINE ===")

        pipeline = TrainingPipeline()
        model_trainer_artifact, model_pusher_artifact = pipeline.run_pipeline()

        print("\n--- Final Results ---")
        print(f"Model Trainer Artifact: {model_trainer_artifact}")
        print(f"Model Pusher Artifact: {model_pusher_artifact}")
        print("---------------------\n")

        logger.info("Local testing completed successfully.")


    except Exception as e:
        raise AirQualityException(e,sys)