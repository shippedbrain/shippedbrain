import os

MODEL_NAME = "Elastic-Net-Test"
MLRUNS_PATH = "/tmp/shippedbrain-test/mlruns"

LOGIN_URL_HTTP_BIN = "http://httpbin.org/post"
UPLOAD_URL_HTTP_BIN = "http://httpbin.org/post"

LOGIN_URL_HTTP = os.getenv("SHIPPED_BRAIN_LOGIN_URL")
UPLOAD_URL_HTTP = os.getenv("SHIPPED_BRAIN_UPLOAD_URL")