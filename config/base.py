import os


class BaseConfig:
    """Base configuration for the cardiovascular risk platform."""

    APP_NAME = "Cardio Cerebrovascular Risk Platform"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    # Local dataset configuration.
    DATASET_FILE_PATH = "data/raw/datasets/cardio_train.csv"
    DATASET_ENCODING = "utf-8"
    DATASET_SEPARATOR = ","

    # Distributed processing configuration.
    DISTRIBUTED_MODE_ENABLED = True
    STAGING_DATA_PATH = "data/staging/cardio_staging.csv"
    FEATURE_DATA_PATH = "data/feature/cardio_features.csv"
    HDFS_INPUT_PATH = "hdfs://uestc04:8020/input/cardio_project/raw/cardio_train.csv"
    HDFS_STAGING_PATH = "hdfs://uestc04:8020/input/cardio_project/staging"
    HDFS_FEATURE_PATH = "hdfs://uestc04:8020/input/cardio_project/feature/cardio_features"
    HDFS_WEB_URL = "http://192.168.174.128:9870"
    HDFS_USER = "zhao"
    HDFS_UPLOAD_DIR = "/input/cardio_project/raw"

    # Database connection placeholders. Fill these values after you provide them.
    MYSQL_HOST = ""
    MYSQL_PORT = 3306
    MYSQL_DATABASE = ""
    MYSQL_USERNAME = ""
    MYSQL_PASSWORD = ""

    REDIS_HOST = ""
    REDIS_PORT = 6379
    REDIS_PASSWORD = ""

    HIVE_METASTORE_URI = ""
    HBASE_HOST = ""
    HBASE_PORT = 9090

    JSON_AS_ASCII = False

    CARDIO_FEATURE_COLUMNS = [
        "age",
        "gender",
        "bmi",
        "cholesterol",
        "diabetes",
        "hypertension",
        "smoker",
        "alcohol",
        "exercise",
    ]
    CARDIO_TARGET_COLUMN = "target_disease"
    TRAIN_TEST_SPLIT_RATIO = 0.2
    RANDOM_STATE = 42

    MODEL_OUTPUT_DIR = "data/feature/models"
    DEFAULT_MODEL_NAME = "logistic_regression"
    AVAILABLE_MODEL_NAMES = ["logistic_regression", "random_forest"]
    ENABLE_MULTI_RUN_TRAINING = True
    DEFAULT_TRAINING_ROUNDS = 3
    BEST_MODEL_METRIC = "roc_auc"
