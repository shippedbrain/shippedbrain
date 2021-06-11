import pytest
import shutil
import os
import tests.resources.train as train
from tests import MLRUNS_PATH

def pytest_sessionstart(session):
    print('\n[STARTUP] Setting mlflow tracking uri')
    try:
        MLFLOW_TRACKING_URI  = f"file://{MLRUNS_PATH}"
        os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        run_id = train.main()
        os.environ.setdefault("SHIPPED_BRAIN_TEST_RUN_ID", run_id.info.run_id)
    except Exception as e:
        print(f"[EXCEPTION] Failed to run startup routine! Error: {e}")

def pytest_sessionfinish(session, exitstatus):
    print("\n[TEARDOWN] Deleting mlruns. Test run_id:", os.environ.get("SHIPPED_BRAIN_TEST_RUN_ID"))
    #print(exitstatus)
    shutil.rmtree(MLRUNS_PATH)
