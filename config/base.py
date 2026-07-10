import os


class BaseConfig:
    """Base configuration for the cardiovascular risk platform."""

    APP_NAME = "Cardio Cerebrovascular Risk Platform"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    # Local dataset configuration.
    DATASET_FILE_PATH = "data/raw/datasets/cardio_train.csv"
    DATASET_ENCODING = "utf-8"
    DATASET_SEPARATOR = ";"

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
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]
    CARDIO_TARGET_COLUMN = "cardio"
    TRAIN_TEST_SPLIT_RATIO = 0.2
    RANDOM_STATE = 42
