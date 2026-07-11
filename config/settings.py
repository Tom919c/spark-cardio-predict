"""Configuration helpers shared by local Flask and optional Spark execution."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    """Load a tiny .env file without adding a runtime dependency."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


class BaseConfig:
    """Runtime settings with safe local defaults."""

    APP_NAME = "CardioSpark"
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret")
    JSON_AS_ASCII = False

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = _as_int("PORT", 5000)
    DEBUG = _as_bool(os.getenv("DEBUG"), True)
    APP_ENV = os.getenv("APP_ENV", "development")

    DATA_MODE = os.getenv("DATA_MODE", "local").strip().lower()
    DISTRIBUTED_MODE_ENABLED = DATA_MODE == "hdfs"
    DATASET_FILE_PATH = _project_path(
        os.getenv("DATASET_FILE_PATH", "data/raw/datasets/cardio_train.csv")
    )
    POPULATION_DATASET_PATH = _project_path(
        os.getenv("POPULATION_DATASET_PATH", "data/dws/chengdu_resident_health_simulated.csv")
    )
    STAGING_DATA_PATH = _project_path("data/staging/cardio_staging.csv")
    FEATURE_DATA_PATH = _project_path("data/feature/cardio_features.csv")
    MODEL_OUTPUT_DIR = _project_path(os.getenv("MODEL_OUTPUT_DIR", "data/feature/models"))
    MODEL_MANIFEST_PATH = _project_path("data/feature/models/active_models.json")
    DATASET_ENCODING = os.getenv("DATASET_ENCODING", "utf-8")
    DATASET_SEPARATOR = os.getenv("DATASET_SEPARATOR", ",")

    HDFS_NAMENODE_URI = os.getenv("HDFS_NAMENODE_URI", "hdfs://localhost:9000")
    HDFS_WEB_URL = os.getenv("HDFS_WEB_URL", "http://localhost:9870")
    HDFS_USER = (
        os.getenv("HDFS_USER")
        or os.getenv("USER")
        or os.getenv("USERNAME")
        or "cardiospark"
    )
    HDFS_RAW_PATH = os.getenv("HDFS_RAW_PATH", f"/user/{HDFS_USER}/cardiospark/raw")
    HDFS_INPUT_PATH = os.getenv(
        "HDFS_INPUT_PATH",
        f"{HDFS_RAW_PATH}/chengdu_resident_health_simulated.csv",
    )
    HDFS_STAGING_PATH = os.getenv(
        "HDFS_STAGING_PATH", f"/user/{HDFS_USER}/cardiospark/staging"
    )
    HDFS_FEATURE_PATH = os.getenv(
        "HDFS_FEATURE_PATH", f"/user/{HDFS_USER}/cardiospark/feature"
    )
    SPARK_SUBMIT_BIN = os.getenv("SPARK_SUBMIT_BIN", "spark-submit")

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
    HEART_LABEL_COLUMN = "label_heart"
    STROKE_LABEL_COLUMN = "label_stroke"
    TRAIN_TEST_SPLIT_RATIO = _as_float("TRAIN_TEST_SPLIT_RATIO", 0.2)
    RANDOM_STATE = _as_int("RANDOM_STATE", 42)
    RANDOM_FOREST_TREES = _as_int("RANDOM_FOREST_TREES", 200)

    RISK_TARGETS = {
        "heart": HEART_LABEL_COLUMN,
        "stroke": STROKE_LABEL_COLUMN,
    }
    RISK_LEVEL_THRESHOLDS = (0.15, 0.35, 0.60, 0.80)

    DEFAULT_HEART_MODEL_FILE = os.getenv("DEFAULT_HEART_MODEL_FILE", "")
    DEFAULT_STROKE_MODEL_FILE = os.getenv("DEFAULT_STROKE_MODEL_FILE", "")

    @classmethod
    def as_dict(cls) -> dict:
        return {
            key: value
            for key, value in cls.__dict__.items()
            if key.isupper() and not key.startswith("_")
        }
