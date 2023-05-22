import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_HOME_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))

DEFAULT_DATA_PATH = os.path.join(DEFAULT_HOME_PATH, "data")
DEFAULT_ARTIFACTS_PATH = os.path.join(DEFAULT_HOME_PATH, "artifacts")
DEFAULT_RESULTS_PATH = os.path.join(DEFAULT_HOME_PATH, "results")

