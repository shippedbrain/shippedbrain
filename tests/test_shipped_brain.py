import json
import pytest
import yaml
import tempfile
from shippedbrain import shippedbrain
import os
import mlflow
from tests import MODEL_NAME, MLRUNS_PATH, LOGIN_URL_HTTP_BIN, UPLOAD_URL_HTTP_BIN, LOGIN_URL_HTTP, UPLOAD_URL_HTTP
import tests.resources.train as train

client = mlflow.tracking.MlflowClient()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RUN_ID = os.environ.get("SHIPPED_BRAIN_TEST_RUN_ID")

MODEL_ARTIFACTS_DIR = "model"
RUN_PATH = os.path.join(MLRUNS_PATH, f"0/{RUN_ID}/")


class TestShippedBrain:

    def test__validate_model_name(self):
        # positive
        assert shippedbrain._validate_model_name("Model")
        assert shippedbrain._validate_model_name("Model-")
        assert shippedbrain._validate_model_name("Mod_el")
        assert shippedbrain._validate_model_name("model2-a")
        # negative
        assert shippedbrain._validate_model_name("Mod/el") is False
        assert shippedbrain._validate_model_name("-Model") is False
        assert shippedbrain._validate_model_name("12model2-a") is False

    def test__is_valid_flavor(self):
        flavors = [
            "pyfunc",
            "h2o",
            "keras",
            "lightgbm",
            "pytorch",
            "sklearn",
            "statsmodels",
            "tensorflow",
            "xgboost",
            "spacy",
            "fastai"
        ]
        assert all([shippedbrain._is_valid_flavor(f) for f in flavors])

    def test__validate_run_id(self,):
        assert shippedbrain._validate_run_id(mlflow_client=client, run_id=RUN_ID), 'Failed to validate run_id.'

    def test__get_logged_model(self):
        logged_model = shippedbrain._get_logged_model(RUN_ID)
        print("LOGGED MODEL:", logged_model)

    def test__validate_model(self):
        assert shippedbrain._validate_run_id(client, RUN_ID), "Failed to validate model."

    def test__get_model_artifacts_path(self):
        model_artifacts_dir = "model"
        model_artifacts_dir_result = shippedbrain._get_model_artifacts_path(RUN_ID)
        assert model_artifacts_dir_result == model_artifacts_dir,\
            f"Failed to get model artifacts path. Return path is '{model_artifacts_dir_result}'"

    def test__download_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shippedbrain._download_artifacts(client, RUN_ID, tmpdir)
            # TODO compare dirs

    def test__create_shipped_brain_yaml(self):
        shipped_brain_yaml = {
            "model_name": MODEL_NAME,
            "flavor": shippedbrain.flavor_name["pyfunc"],
            "model_artifacts_path": MODEL_ARTIFACTS_DIR
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            shipped_brain_yaml_path = os.path.join(tmpdir, "shipped-brain.yaml")

            shipped_brain_yaml_path_result = shippedbrain._create_shipped_brain_yaml(MODEL_NAME,
                                                                                     MODEL_ARTIFACTS_DIR,
                                                                                     shippedbrain.flavor_name["pyfunc"],
                                                                                     tmpdir)

            assert shipped_brain_yaml_path_result == shipped_brain_yaml_path

            with open(shipped_brain_yaml_path, "r") as file:
                shipped_brain_yaml_result = yaml.full_load(file)

                assert shipped_brain_yaml_result == shipped_brain_yaml, \
                    f"Bad shipped-brain.yaml file content! Content: {shipped_brain_yaml_result}"

    def test__update_MLmodel(self):
        import shutil
        from datetime import datetime

        mlmodel_path = './tests/resources/data/MLmodel'
        time_now = datetime.utcnow()
        yaml_base = dict()
        with open(mlmodel_path, "r") as file:
            yaml_base = yaml.full_load(file)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(mlmodel_path, tmpdir)
            shippedbrain._update_MLmodel(RUN_ID, tmpdir, "", utc_time_created=time_now)
            created_yaml_path = os.path.join(tmpdir, "MLmodel")

            with open(created_yaml_path, "r") as file:
                yaml_created = yaml.full_load(file)

                assert yaml_created.pop("run_id") == RUN_ID
                assert yaml_created.pop("utc_time_created") == str(time_now)
                yaml_base.pop("run_id")
                yaml_base.pop("utc_time_created")
                assert yaml_created == yaml_base

    def test__zip_artifacts(self):
        zip_file_path = shippedbrain._zip_artifacts(RUN_PATH)
        assert os.path.isfile(zip_file_path)

    def test__unzip_artifacts(self):
        zipfile_path = "./tests/resources/data/zipfile_example.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            shippedbrain._unzip_artifacts(zipfile_path, tmpdir)
            unzippedfile_path = os.path.join(tmpdir, "MLmodel")

            assert os.path.isfile(unzippedfile_path)
            # TODO read content ...

    # INTEGRATION
    def test__upload_file(self):
        import json
        zipfile_path = "./tests/resources/data/zipfile_example.zip"
        bearer = "some-auth-bearer"
        # Uses httpbin by default
        response = shippedbrain._upload_file(zipfile_path, bearer, UPLOAD_URL_HTTP_BIN)

        assert  response.status_code == 200
        response_json = json.loads(response.text)
        assert response_json["headers"]["Authorization"] == f"Bearer {bearer}"

    # INTEGRATION
    def test__login(self):
        email = "shippedbrain@mail.com"
        password = "passowrd"

        response = shippedbrain._login(email, password, LOGIN_URL_HTTP_BIN)
        response_text_as_dict = eval(response.text)
        response_data = eval(response_text_as_dict["data"])

        assert response.status_code == 200
        assert response_data["email"] == email
        assert response_data["password"] == password

    def test__log_model(self):
        artifacts_base = client.list_artifacts(RUN_ID)

        artifacts_path = os.path.join(RUN_PATH, "artifacts")
        logged_run = shippedbrain._log_model(artifacts_path, MODEL_ARTIFACTS_DIR)
        assert shippedbrain._validate_run_id(client, logged_run.info.run_id)
        artifacts_result = client.list_artifacts(logged_run.info.run_id)
        assert artifacts_base == artifacts_result

    def test__log_flavor_inside_run_context(self):
        log_model_option = {"flavor": "_log_flavor"}

        run = train.main(log_model_option=log_model_option, run_inside_mlflow_context=True)

        assert shippedbrain._validate_run_id(client, run.info.run_id), "Run id is not valid"
        assert shippedbrain._validate_model(run.info.run_id), "Logged model is not valid"

    def test__log_flavor_outside_run_context(self):
        log_model_option = {"flavor": "_log_flavor"}

        run = train.main(log_model_option=log_model_option, run_inside_mlflow_context=False)

        assert shippedbrain._validate_run_id(client, run.info.run_id), "Run id is not valid"
        assert shippedbrain._validate_model(run.info.run_id), "Logged model is not valid"

    def test_upload_run(self):
        response = shippedbrain.upload_run(run_id=RUN_ID,
                                           model_name="Test-Model-upload_run",
                                           email="bernardo@shippedbrain.com",
                                           password="password",
                                           login_url=LOGIN_URL_HTTP,
                                           upload_url=UPLOAD_URL_HTTP)

        assert response.status_code == 200

    def test_upload_model(self):
        train_x, train_y, test_x, test_y, alpha, l1_ratio = train.build_train()

        lr, predicted_qualities, signature, rmse, mae, r2 = train.train_and_eval(alpha,
                                                                                  l1_ratio,
                                                                                  train_x,
                                                                                  train_y,
                                                                                  test_x,
                                                                                  test_y)
        input_example = test_x.iloc[0:2]

        MAX_RETRIES = 3
        curr_try = 0
        import  time
        while curr_try < MAX_RETRIES:
            curr_try += 1
            response = shippedbrain.upload_model(flavor="sklearn",
                                                 model_name="Test-Model-upload_model",
                                                 email="bernardo@shippedbrain.com",
                                                 password="password",
                                                 signature=signature,
                                                 input_example=input_example,
                                                 sk_model=lr,
                                                 artifact_path="model",
                                                 login_url=LOGIN_URL_HTTP,
                                                 upload_url=UPLOAD_URL_HTTP
                                                 )
            # FORBIDDEN, server upload concurrency restrictions
            if response.status_code == 403:
                time.sleep(2 ** curr_try)
            else:
                break
        assert response.status_code == 200

