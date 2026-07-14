"""项目配置：当前运行环境中的 Flask/Spark 与 VMware HDFS 共用的解析层。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    """加载 .env 文件（不引入 python-dotenv 依赖）。"""
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
    """运行时配置，clone 后开箱即用的安全默认值。"""

    APP_NAME = "CardioSpark"
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret")
    JSON_AS_ASCII = False

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = _as_int("PORT", 5000)
    DEBUG = _as_bool(os.getenv("DEBUG"), True)
    APP_ENV = os.getenv("APP_ENV", "development")

    # 默认本地模式，未启动 HDFS 时 Flask 仍可用于演示和接口测试。
    DATA_MODE = os.getenv("DATA_MODE", "local").strip().lower()
    DISTRIBUTED_MODE_ENABLED = DATA_MODE == "hdfs"
    DATASET_FILE_PATH = _project_path(
        os.getenv("DATASET_FILE_PATH", "data/raw/CVD_Standard_DWD_refined.csv")
    )
    POPULATION_DATASET_PATH = _project_path(
        os.getenv(
            "POPULATION_DATASET_PATH",
            "data/raw/chengdu_resident_health_simulated.csv",
        )
    )
    STAGING_DATA_PATH = _project_path("data/raw/CVD_Standard_DWD.csv")
    FEATURE_DATA_PATH = _project_path("data/features/cardio_features.csv")
    MODEL_OUTPUT_DIR = _project_path(os.getenv("MODEL_OUTPUT_DIR", "data/models"))
    MODEL_MANIFEST_PATH = _project_path(
        os.getenv("MODEL_MANIFEST_PATH", "data/models/active_models.json")
    )
    TRAINING_RESULTS_PATH = _project_path(
        os.getenv("TRAINING_RESULTS_PATH", "docs/training_results.md")
    )
    DATASET_ENCODING = os.getenv("DATASET_ENCODING", "utf-8")
    DATASET_SEPARATOR = os.getenv("DATASET_SEPARATOR", ",")

    # 阶段二数据集、任务和本地开发配置。
    DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite").strip().lower()
    DATABASE_PATH = _project_path(
        os.getenv("DATABASE_PATH", "data/localstorage/app.db")
    )
    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = _as_int("MYSQL_PORT", 3306)
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "cardiospark")
    UPLOAD_ROOT = _project_path(
        os.getenv("UPLOAD_ROOT", "data/localstorage/uploads")
    )
    LOCAL_RESULT_ROOT = _project_path(
        os.getenv("LOCAL_RESULT_ROOT", "data/localstorage/results")
    )
    UPLOAD_MAX_FILE_SIZE = _as_int("UPLOAD_MAX_FILE_SIZE", 512 * 1024 * 1024)
    UPLOAD_CHUNK_SIZE = _as_int("UPLOAD_CHUNK_SIZE", 8 * 1024 * 1024)
    LOCAL_ANALYSIS_CHUNK_SIZE = _as_int("LOCAL_ANALYSIS_CHUNK_SIZE", 100_000)
    TASK_WORKERS = _as_int("TASK_WORKERS", 2)
    PRIVACY_MIN_GROUP_SIZE = _as_int("PRIVACY_MIN_GROUP_SIZE", 5)
    KNOWLEDGE_BASE_PATH = _project_path(
        os.getenv(
            "KNOWLEDGE_BASE_PATH", "resources/knowledge/intervention_rules.json"
        )
    )
    MAP_ASSET_DIR = _project_path(os.getenv("MAP_ASSET_DIR", "static/geo"))
    SPARK_ANALYSIS_SCRIPT = _project_path(
        os.getenv("SPARK_ANALYSIS_SCRIPT", "scripts/run_phase2_analysis.py")
    )
    # Spark 在当前应用所在环境中原生执行，VMware Ubuntu 提供 HDFS 存储。
    SPARK_SCORE_ENGINE = os.getenv("SPARK_SCORE_ENGINE", "pandas").strip().lower()

    # DATA_MODE=hdfs 时使用以下 HDFS 参数；地址由每位成员在 .env 中填写。
    HDFS_NAMENODE_URI = os.getenv("HDFS_NAMENODE_URI", "hdfs://localhost:9000")
    HDFS_WEB_URL = os.getenv("HDFS_WEB_URL", "http://localhost:9870")
    # WebHDFS 重定向到不可达主机名时，在本机 .env 中覆盖为可达地址。
    HDFS_DATANODE_HOST = os.getenv("HDFS_DATANODE_HOST", "")
    # 默认使用当前系统用户，团队成员可在 .env 中覆盖为 Hadoop 用户名。
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
    HDFS_UPLOAD_ROOT = os.getenv(
        "HDFS_UPLOAD_ROOT", f"{HDFS_RAW_PATH}/_uploads"
    )
    HDFS_RESULT_ROOT = os.getenv(
        "HDFS_RESULT_ROOT", f"{HDFS_FEATURE_PATH}/ads"
    )
    SPARK_SUBMIT_BIN = os.getenv("SPARK_SUBMIT_BIN", "spark-submit")
    SPARK_PYTHON = os.getenv("SPARK_PYTHON", sys.executable)

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
    RANDOM_STATE = _as_int("RANDOM_STATE", 20260714)
    XGBOOST_TREES = _as_int("XGBOOST_TREES", 850)
    XGBOOST_MAX_DEPTH = _as_int("XGBOOST_MAX_DEPTH", 5)
    XGBOOST_LEARNING_RATE = _as_float("XGBOOST_LEARNING_RATE", 0.05)
    XGBOOST_MIN_CHILD_WEIGHT = _as_float("XGBOOST_MIN_CHILD_WEIGHT", 6.0)
    XGBOOST_SUBSAMPLE = _as_float("XGBOOST_SUBSAMPLE", 0.85)
    XGBOOST_COLSAMPLE_BYTREE = _as_float("XGBOOST_COLSAMPLE_BYTREE", 0.90)
    XGBOOST_GAMMA = _as_float("XGBOOST_GAMMA", 0.0)
    XGBOOST_REG_ALPHA = _as_float("XGBOOST_REG_ALPHA", 0.0)
    XGBOOST_REG_LAMBDA = _as_float("XGBOOST_REG_LAMBDA", 2.0)
    STROKE_MIN_RECALL = _as_float("STROKE_MIN_RECALL", 0.70)

    # 阶段一固定为两个独立事件模型，不在此处扩展额外疾病类型。
    RISK_TARGETS = {
        "heart": HEART_LABEL_COLUMN,
        "stroke": STROKE_LABEL_COLUMN,
    }
    RISK_LEVEL_THRESHOLDS = (0.15, 0.35, 0.60, 0.80)

    @classmethod
    def as_dict(cls) -> dict:
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if key.isupper() and not key.startswith("_")
        }
