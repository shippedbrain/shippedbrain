# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

"""
CREATE LOCAL mlruns
"""

import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from  tests import MODEL_NAME

import logging

from shippedbrain import shippedbrain

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_eval(alpha, l1_ratio, train_x, train_y, test_x, test_y):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=46)

    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Infer model signature
    signature = infer_signature(test_x, predicted_qualities)

    return lr, predicted_qualities, signature, rmse, mae, r2

def build_train():
    # Read the wine-quality csv file from the URL
    # data = pd.read_csv("./tests/resources/data/winequality-red.csv", sep=",", header=True)
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        print("Downloading dataset...")
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = 0.5
    l1_ratio = 0.5

    return train_x, train_y, test_x, test_y, alpha, l1_ratio

def main(log_model_option: dict = {"flavor": "mlflow"}, run_inside_mlflow_context: bool = True):
    """ Train and log model

    :param log_model_option: Log model using options:
                             if {"flavor": "mlflow"}: log model using mlflow log_model method
                             else if {"flavor": "_log_model"} log model using shipped brain _log_model function
                             else ig {"flavor": "upload_run" | "upload_model", args...} log model using named function with args
                             NB: input_example and signature are not required
    :param run_inside_mlflow_context: if True run log method from mlflow run context,
                                      otherwise use shippedbrain.log_flavor outside without mlflow run context
    """
    warnings.filterwarnings("ignore")
    np.random.seed(46)
    

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    train_x, train_y, test_x, test_y, alpha, l1_ratio = build_train()

    lr, predicted_qualities, signature, rmse, mae, r2 = train_and_eval(alpha,
                                                                       l1_ratio,
                                                                       train_x,
                                                                       train_y,
                                                                       test_x,
                                                                       test_y)

    log_model_option["signature"] = signature
    log_model_option["input_example"] = test_x.iloc[0:2]

    print(f"[INFO] RUN INSIDE MLFLOW RUN CONTEXT={run_inside_mlflow_context}")
    if run_inside_mlflow_context:
        with mlflow.start_run() as run:
            print("[INFO] Starting run with id:", run.info.run_id)

            print("[INFO]Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("[INFO]\tRMSE: %s" % rmse)
            print("[INFO]\tMAE: %s" % mae)
            print("[INFO]\tR2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            print("[DEBUG] Log model option flavor:", log_model_option["flavor"])
            if log_model_option["flavor"] == "_log_flavor":
                _ = shippedbrain._log_flavor("sklearn", sk_model = lr, signature = signature, input_example = log_model_option["input_example"], artifact_path="model")
            # INTEGRATION
            elif log_model_option["flavor"] == "upload_model" or log_model_option["flavor"] == "upload_run":
                flavor = log_model_option["flavor"]
                log_model_option.pop("flavor")
                log_func = eval(f"shippedbrain.{flavor}")
                _ = log_func(**log_model_option)
            elif log_model_option["flavor"] == "mlflow":
                mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=log_model_option["input_example"])

            print(f"[INFO] Model URI runs:/{run.info.run_id}/model\n")

        return run

    else:
        if log_model_option["flavor"] == "_log_flavor":
            run = shippedbrain._log_flavor("sklearn",
                                           sk_model=lr,
                                           signature=signature,
                                           input_example=log_model_option["input_example"],
                                           artifact_path="model")
        # INTEGRATION
        elif log_model_option["flavor"] == "upload_model" or log_model_option["flavor"] == "upload_run":
            flavor = log_model_option["flavor"]
            log_model_option.pop("flavor")
            log_func = eval(f"shippedbrain.{flavor}")
            run = log_func(**log_model_option)

        print(f"[INFO] Model URI runs:/{run.info.run_id}/model\n")

        return run
