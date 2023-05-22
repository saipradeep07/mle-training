import os

# data pull, split
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

test_size = 0.2


# paths
log_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "logs/run_logger.log",
)
