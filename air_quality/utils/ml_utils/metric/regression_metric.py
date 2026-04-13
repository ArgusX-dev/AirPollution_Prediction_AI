from air_quality.exception.exception import AirQualityException
from air_quality.entity.artifact_entity import RegressionMetricArtifact
import sys
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return RegressionMetricArtifact(rmse_score=rmse, mae_score=mae, r2_score=r2)
    except Exception as e:
        raise AirQualityException(e, sys)