import yaml
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger
import os,sys
import dill
import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,'rb') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise AirQualityException(e,sys)


def write_yaml_file(file_path:str, content:object,replace:bool = False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as h:
            yaml.dump(content,h)
    except Exception as e:
        raise AirQualityException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as h:
            np.save(h,array)
    except Exception as e:
        raise AirQualityException(e,sys)

def save_object(file_path:str,obj:object)-> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as z:
            pickle.dump(obj,z)
    except Exception as e:
        raise AirQualityException(e,sys)



def load_object(file_path:str)-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} does not exist")
        with open(file_path,'rb') as x:
            print(x)
            return pickle.load(x)
    except Exception as e:
        raise AirQualityException(e,sys)

def load_numpy_array_data(file_path:str)-> np.array:
    try:
        with open(file_path,'rb') as g:
            return np.load(g)
    except Exception as e:
        raise AirQualityException(e,sys)


def optimize_xgboost_hyperparameters(x_train, y_train):
    try:
        split_idx = int(len(x_train) * 0.8)
        x_tr, x_val = x_train[:split_idx], x_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1
            }

            model = XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

            y_pred = model.predict(x_val)

            return r2_score(y_val, y_pred)


        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        study.optimize(objective, n_trials=30)

        return study.best_params

    except Exception as e:
        raise AirQualityException(e, sys)