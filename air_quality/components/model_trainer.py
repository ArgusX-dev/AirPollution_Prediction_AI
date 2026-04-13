import os
import sys
from xgboost import XGBRegressor
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
from air_quality.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from air_quality.entity.config_entity import ModelTrainerConfig
from air_quality.utils.main_utils.utils import load_numpy_array_data, load_object, save_object, optimize_xgboost_hyperparameters
from air_quality.utils.ml_utils.metric.regression_metric import get_regression_score
from air_quality.utils.ml_utils.model.estimator import AirModel
import mlflow
import dagshub




class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise AirQualityException(e, sys)

    def track_mlflow(self, best_model, best_params, train_metric, test_metric):
        dagshub.auth.add_app_token(token=os.environ["DAGSHUB_USER_TOKEN"])
        dagshub.init(repo_owner='ArgusX', repo_name='AirPollution_Prediction_AI', mlflow=True)
        try:
            mlflow.set_experiment("Air_Quality_Risk_Severity")

            with mlflow.start_run():

                mlflow.log_params(best_params)

                mlflow.log_metric('train_rmse', train_metric.rmse_score)
                mlflow.log_metric('train_r2', train_metric.r2_score)
                mlflow.log_metric('train_mae', train_metric.mae_score)

                mlflow.log_metric('test_rmse', test_metric.rmse_score)
                mlflow.log_metric('test_r2', test_metric.r2_score)
                mlflow.log_metric('test_mae', test_metric.mae_score)

                mlflow.xgboost.log_model(best_model, "xgboost_model")

                logger.info("Experiment successfully registered in MLflow.")

        except Exception as e:
            raise AirQualityException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            logger.info("Starting hyperparameter optimization with Optuna (TPE)...")
            best_params = optimize_xgboost_hyperparameters(x_train, y_train)
            logger.info(f"Best hyperparameters found: {best_params}")

            best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            logger.info("Training the final XGBoost model...")
            best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            regression_train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            logger.info(f"Metrics Train -> RMSE: {regression_train_metric.rmse_score:.4f}, R2: {regression_train_metric.r2_score:.4f}")


            y_test_pred = best_model.predict(x_test)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
            logger.info(f"Metrics Test -> RMSE: {regression_test_metric.rmse_score:.4f}, R2: {regression_test_metric.r2_score:.4f}")

            self.track_mlflow(
                best_model=best_model,
                best_params=best_params,
                train_metric=regression_train_metric,
                test_metric=regression_test_metric
            )

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            air_model = AirModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=air_model)

            save_object('final_model/model.pkl',best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric
            )

            return model_trainer_artifact

        except Exception as e:
            raise AirQualityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("=== STARTING MODEL TRAINER ===")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logger.info("=== MODEL TRAINER COMPLETED ===")
            return model_trainer_artifact
        except Exception as e:
            raise AirQualityException(e, sys)