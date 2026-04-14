from unittest.mock import patch, MagicMock
from air_quality.components.model_trainer import ModelTrainer
from air_quality.components.model_pusher import ModelPusher
import numpy as np

@patch('air_quality.components.model_trainer.mlflow')
@patch('air_quality.components.model_trainer.dagshub')
@patch('air_quality.components.model_trainer.optimize_xgboost_hyperparameters')
@patch('air_quality.components.model_trainer.XGBRegressor')
def test_model_trainer_flow(mock_xgb, mock_optimize, mock_dagshub, mock_mlflow):

    mock_optimize.return_value = {"n_estimators": 10, "max_depth": 3}
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = np.zeros(50)
    mock_xgb.return_value = mock_model_instance
    # -----------------------------------

    mock_config = MagicMock()
    mock_artifact = MagicMock()

    x_dummy = np.random.rand(50, 5)
    y_dummy = np.random.rand(50)

    trainer = ModelTrainer(mock_config, mock_artifact)

    with patch('air_quality.components.model_trainer.os.environ', {"DAGSHUB_USER_TOKEN": "dummy"}), \
         patch('air_quality.components.model_trainer.load_object', return_value=MagicMock()), \
         patch('air_quality.components.model_trainer.save_object'):

        with patch.object(trainer, 'track_mlflow') as mock_track:
            artifact = trainer.train_model(x_dummy, y_dummy, x_dummy, y_dummy)

            assert mock_xgb.called
            assert mock_track.called
            assert artifact is not None


@patch('air_quality.components.model_pusher.boto3')
def test_model_pusher_s3_upload(mock_boto3):
    mock_s3_client = MagicMock()
    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.return_value = {'Arn': 'arn:aws:iam::123:user/testing'}

    def boto_client_side_effect(service_name):
        if service_name == 's3': return mock_s3_client
        if service_name == 'sts': return mock_sts_client

    mock_boto3.client.side_effect = boto_client_side_effect

    mock_artifact = MagicMock()
    mock_config = MagicMock()
    mock_config.bucket_name = "tests-bucket"
    mock_config.s3_model_key_path = "models/v1"

    pusher = ModelPusher(mock_artifact, mock_config)

    with patch('air_quality.components.model_pusher.os.path.exists', return_value=True):
        pusher.initiate_model_pusher()

        assert mock_s3_client.upload_file.call_count == 2