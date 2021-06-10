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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# if __name__ == "__main__":
def main():
    warnings.filterwarnings("ignore")
    np.random.seed(46)
    

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    # Read the wine-quality csv file from the URL
    # data = pd.read_csv("./tests/resources/data/winequality-red.csv", sep=",", header=True)
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
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

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run() as run:
        print("[INFO] Starting run with id:", run.info.run_id)
    
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=46)
        
        print("[INFO] Training...")
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("[INFO]Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("[INFO]\tRMSE: %s" % rmse)
        print("[INFO]\tMAE: %s" % mae)
        print("[INFO]\tR2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Infer model signature
        signature = infer_signature(test_x, predicted_qualities)

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name=MODEL_NAME, signature=signature, input_example=test_x.iloc[0:2])
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=test_x.iloc[0:2])
        
        print(f"[INFO] Model URI runs:/{run.info.run_id}/model\n")

    return run
