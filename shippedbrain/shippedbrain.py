import mlflow
from mlflow.entities.run_status import RunStatus
import yaml
from datetime import datetime
import shutil
import os
import tempfile
from typing import Optional
import requests
import json
import uuid
import re
from shippedbrain import LOGIN_URL, UPLOAD_URL

flavors = {
    "pyfunc": mlflow.pyfunc,
    "h2o": mlflow.h2o,
    "keras": mlflow.keras,
    "lightgbm": mlflow.lightgbm,
    "pytorch": mlflow.pytorch,
    "sklearn": mlflow.sklearn,
    "statsmodels": mlflow.statsmodels,
    "tensorflow": mlflow.tensorflow,
    "xgboost": mlflow.xgboost,
    "spacy": mlflow.spacy,
    "fastai": mlflow.fastai
}

# Key in MLmodel file
flavor_name = {
    "pyfunc": "python_function",
    "h2o": "h2o",
    "keras": "keras",
    "lightgbm": "lightgbm",
    "pytorch": "pytorch",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "tensorflow": "tensorflow",
    "xgboost": "xgboost",
    "spacy": "spacy",
    "fastai": "fastai"
}

valid_model_name_regex = r"^[a-zA-Z]+[\w-]*$"

def _validate_model_name(model_name: str) -> bool:
    """ Validate model name - match with regex

    :param model_name: name of the model to publish on app.shippedbrain.com

    :return: True if name is valid, False otherwise
    """
    return type(model_name) is str and re.match(valid_model_name_regex, model_name) is not None


def _validate_run_id(mlflow_client: mlflow.tracking.MlflowClient,
                     run_id: str) -> bool:
    """ Validate user's run from existing model

    :param mlflow_client: MlflowClient instance
    :param run_id: run id of logged mlflow model

    :return: True if run id is valid, False otherwise
    """
    try:
        run_info = mlflow_client.get_run(run_id)

        if RunStatus.from_string(run_info.info.status) == mlflow.entities.run_status.RunStatus.FINISHED:
            return True

        return False
    except:
        raise Exception(f"Failed to get run with id {run_id}")


def _get_logged_model(run_id: str) -> dict:
    """ Get logged model from run id

    :param run_id: mlflow run id of the logged model

    :return: logged model py dict
    """
    _run = mlflow.get_run(run_id)
    # eval is needed, otherwise type is str
    log_model_history = eval(_run.data.to_dictionary()["tags"]["mlflow.log-model.history"])
    for logged_model in log_model_history:
        if run_id == logged_model['run_id']:
            return logged_model

    raise Exception(f"Could not find model with run_id={run_id}")


def _validate_model(run_id: str) -> bool:
    """ Validate logged mlflow model - look for input_example and signature

    :param run_id: the model's run id

    :return: True if model is valid, False otherwise
    """
    logged_model = _get_logged_model(run_id)
    # signature can be inferred from input_example
    try:
        _ = logged_model['signature']
    except Exception as e:
        raise Exception(
            f"Could not fetch signature for model with run id '{run_id}'. Please log a model with a valid "
            f"signature.\nLogged model {logged_model} "
        )
    try:
        _ = logged_model['saved_input_example_info']
    except Exception as e:
        raise Exception(
            f"Could not fetch input example for model with run id '{run_id}'. Please log a model with a valid input "
            f"example.\nLogged model {logged_model}")

    return True


def _get_model_artifacts_path(run_id: str):
    try:
        logged_model = _get_logged_model(run_id)
        return logged_model['artifact_path']
    except:
        raise Exception(f"Could not find model artifact for run_id={run_id}")


def _download_artifacts(mlflow_client: mlflow.tracking.MlflowClient,
                        run_id: str,
                        tmpdirname: str) -> None:
    """Download artifacts from run

    :param mlflow_client: MlflowClient instance
    :param run_id: model's run id
    :param tmpdirname: a tempfile.TempDir path

    :return: None
    """
    model_artifacts = _get_model_artifacts_path(run_id)
    mlflow_client.download_artifacts(run_id=run_id, path=model_artifacts, dst_path=tmpdirname)


def _create_shipped_brain_yaml(model_name: str, model_artifacts_path: str, flavor: str, target_dir) -> str:
    """ Create a shipped-brain.yaml file in models artifacts

    :param model_name: the model's name on app.shippedbrain.com
    :param model_artifacts_path: the mlflow model's artifact path
    :param flavor: the flavor of the model (e.g. "pyfunc")
    :param target_dir: the directory to which to write the shipped-brain.yaml file

    :return: shipped-brain.yaml file absolute path
    """
    shipped_brain_yaml_file = os.path.join(target_dir, "shipped-brain.yaml")
    shipped_brain_yaml = {"model_name": model_name,
                          "model_artifacts_path": model_artifacts_path,
                          "flavor": flavor}
    with open(shipped_brain_yaml_file, "w") as yaml_file:
        yaml.dump(shipped_brain_yaml, yaml_file)

    return shipped_brain_yaml_file

def _update_MLmodel(new_run_id: str,
                    artifacts_uri: str,
                    model_artifacts_dir: str,
                    utc_time_created: Optional[datetime] = None) -> None:
    """ Update run_id field in MLmodel file

    :param new_run_id: new run id to update MLmodel
    :param artifacts_uri: artifacts path
    :param model_artifacts_dir: model artifacts dir in [artifacts_uri]
    :param utc_time_created: for tests only datetime to update in MLmodel

    :return: None
    """
    assert utc_time_created is None or type(utc_time_created) == datetime

    model_articats_path = os.path.join(artifacts_uri, model_artifacts_dir)
    mlmodel_path = os.path.join(model_articats_path, "MLmodel")

    with open(mlmodel_path, 'r+') as mlmodel_file:
        mlmodel_yaml = yaml.full_load(mlmodel_file)
        mlmodel_yaml['run_id'] = new_run_id
        mlmodel_yaml['utc_time_created'] = str(datetime.utcnow()) if not utc_time_created else str(utc_time_created)

    with open(mlmodel_path, 'w') as mlmodel_file:
        _ = yaml.dump(mlmodel_yaml, mlmodel_file)


def _zip_artifacts(model_artifacts_path: str) -> str:
    """ Zip artifacts from mlflow run

    :param model_artifacts_path: directory with model artifacts

    :return: temporary zip file name
    """
    file_name = str(uuid.uuid4())
    file_path = os.path.join(model_artifacts_path, file_name)
    shutil.make_archive(file_path, 'zip', model_artifacts_path)

    return file_path + ".zip"


def _unzip_artifacts(zipfile: str, target_dir: str) -> None:
    """ Unzip model artifacts file

    :param zipfile: absolute path to zip file
    :param target_dir: target directory to unpack file to

    :return: None
    """
    archive_format = "zip"

    file_name = zipfile.split("/")[-1]

    # Unpack the archive file
    shutil.unpack_archive(zipfile, target_dir, archive_format)

    # Delete zip file from target dir.
    try:
        os.remove(os.path.join(target_dir, file_name))
    except:
        print(f"Could not remove file {os.path.join(target_dir, file_name)} - target_dir='{target_dir}' file_name='{file_name}")

    print("Archive file unpacked successfully.")


def _login(email: str, password: str, login_url: str = LOGIN_URL) -> requests.Response:
    """ Login to Shipped Brain

    :param email: shipped brain account email
    :param password: shipped brain account password
    :param login_url: login url

    :return: requests.Response with auth token
    """

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = json.dumps({"email": email, "password": password})

    response = requests.post(login_url, headers=headers, data=data)

    return response


def _upload_file(file_path: str,
                 auth_bearer: str,
                 upload_url: str = UPLOAD_URL) -> requests.Response:
    """Upload file request to Shipped Brain

    :param file_path: absolute path to zipfile to upload to app.shippedbrain.com
    :param auth_bearer: valid app.shippedbrain.com Authorization Bearer
    :param upload_url: file upload target url

    :return: upload response - request.Response
    """
    headers = {
        "Authorization": f"Bearer {auth_bearer}"
    }

    filename = file_path.split("/")[-1]

    with open(file_path, "rb") as file_obj:
        file = {
            "file": (filename, file_obj, "multipart/form-data"),
        }

        response = requests.post(upload_url, headers=headers, files=file)

    return response


def _is_valid_flavor(flavor: str) -> bool:
    """ Check if model flavor is valid

    :param flavor: the model flavor (e.g. "pyfunc")

    :return: True if valid flavor, False otherwise
    """
    return flavors.get(flavor) is not None


def _log_model(artifacts_path: str, model_artifacts_dir: str) -> mlflow.entities.Run:
    """ Log model from MLmodel file

    :param artifacts_path: absolute artifact dir. path
    :param model_artifacts_dir: relative model artifacts path in [tmpdir]

    :return: mlflow.entities.Run
    """

    with mlflow.start_run() as run:
        _update_MLmodel(run.info.run_id, artifacts_path, model_artifacts_dir)
        # USE mlflow.log_artifacts, otherwise root dir. is also copied
        mlflow.log_artifacts(artifacts_path)

    return run


def upload(run_id: str,
           model_name: str,
           email: Optional[str] = None,
           password: Optional[str] = None,
           flavor: str = "pyfunc",
           login_url: str = LOGIN_URL,
           upload_url: str = UPLOAD_URL) -> requests.Response:
    """ Upload artifacts to Shipped Brain

    :param run_id: run id of mlflow logged model
    :param model_name: name of the model to publish on app.shippedbrain.com
    :param email: shipped brain account email
    :param password: shipped brain account password
    :param flavor: flavor of the logged model
    :param login_url: login url
    :param upload_url: upload_url

    :return: model upload requests.Response from app.shippedbrain.com
    """

    client = mlflow.tracking.MlflowClient()

    email = email if email else os.getenv("SHIPPED_BRAIN_EMAIL")
    password = password if password else os.getenv("SHIPPED_BRAIN_PASSWORD")

    assert email, f"Bad email. Environment variable SHIPPED_BRAIN_EMAIL is not defined or you did not provide the email argument."
    assert password,f"Bad password. Environment variable SHIPPED_BRAIN_PASSWORD is not defined or you did not provide the password argument."
    assert _validate_model_name(model_name), f"Bad model name '{model_name}! Please provide a valid model name"
    assert _validate_run_id(client, run_id), f"The run id '{run_id}' you provided is not valid!"
    assert _validate_model(run_id), f"The model with '{run_id}' is not valid!"

    model_artifacts_path = _get_model_artifacts_path(run_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        _download_artifacts(client, run_id, tmpdir)
        _create_shipped_brain_yaml(model_name, model_artifacts_path, flavor_name[flavor], tmpdir)
        zipped_file = _zip_artifacts(tmpdir)

        login_response = _login(email, password, login_url)
        assert login_response.status_code == 200, \
            f"Failed to login with status code {login_response.status_code}. Error: {login_response.text}"
        response_data = eval(login_response.text)

        auth_bearer = response_data["data"]["results"]["access_token"]

        upload_response = _upload_file(zipped_file, auth_bearer, upload_url)

        return upload_response

# WiP
# TODO add named args
# def _log_flavor(model: Any,
#                 flavor: str,
#                 conda_env: str,
#                 input_example: Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list],
#                 signature: mlflow.models.signature.ModelSignature):
#     """ Log model flavor to Shipped Brain - similar to mlflow.<flavour>.log_model
#
#     :param model: path to serialized model
#     :param flavor: a valid model flavor
#     :param conda_env: path to conda environment file
#     :param input_example: path to input example file
#     :param signature: an mlflow.Signature
#
#     :return:
#     """
#     assert _is_valid_flavor(flavor), f"Failed to log model. Bad model flavor '{flavor}'."
#
#     # log local e enviar run VS enviar logo para o server
#     # log local é mais fácil
#     # model_logger = flavors[flavor]
#     # with mlflow.start_run() as model_log_run:
#     #     print(f"Logging local model flavor '{flavor}' with run_id={model_log_run.info.run_id}")
#     #     model_logger.log_model(model, conda_env=conda_env, input_example=input_example, signature=signature)
#     #
#     # return model_log_run
#     pass

