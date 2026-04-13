from air_quality.exception.exception import AirQualityException
import os,sys
from air_quality.logging.logger import logger
from air_quality.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import numpy as np



class AirModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise AirQualityException(e,sys)

    def predict(self,X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_hat_raw = self.model.predict(X_transform)
            y_hat_final = np.clip(np.round(y_hat_raw), 1, 5).astype(int)
            return y_hat_final
        except Exception as e:
            raise AirQualityException(e,sys)