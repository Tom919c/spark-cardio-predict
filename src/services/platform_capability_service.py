"""只读的平台运行能力快照。"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.dao.hdfs_dao import HDFSDao
from src.ml.model_registry import ModelRegistry


class PlatformCapabilityService:
    """报告当前部署真正选择的存储、计算和模型能力。"""

    def __init__(self, config):
        self.config = config

    def inspect(self) -> dict:
        data_mode = str(self.config.get("DATA_MODE", "local")).strip().lower()
        database_type = str(self.config.get("DATABASE_TYPE", "sqlite")).strip().lower()
        hdfs = self._hdfs_status(data_mode)
        spark = self._spark_status(data_mode)
        models = self._model_status()
        infrastructure_ready = (
            data_mode == "hdfs" and hdfs["client_ready"] and spark["ready"]
        )
        return {
            "database": {
                "type": database_type,
                "supported_types": ["sqlite", "mysql"],
                "role": "metadata_and_task_state",
            },
            "data_mode": data_mode,
            "storage_engine": "hdfs" if data_mode == "hdfs" else "local_filesystem",
            "compute_engine": "apache_spark" if data_mode == "hdfs" else "local_pandas",
            "hdfs": hdfs,
            "spark": spark,
            "models": models,
            "distributed_infrastructure_ready": infrastructure_ready,
            "distributed_pipeline_ready": infrastructure_ready and models["ready"],
        }

    def _hdfs_status(self, data_mode: str) -> dict:
        dao = HDFSDao.from_config(self.config)
        hadoop_cli = shutil.which("hadoop")
        hdfs_cli = shutil.which("hdfs")
        return {
            "enabled": data_mode == "hdfs",
            "client_ready": dao.is_available(),
            "web_url_configured": bool(str(self.config.get("HDFS_WEB_URL", "")).strip()),
            "namenode_uri_configured": bool(
                str(self.config.get("HDFS_NAMENODE_URI", "")).strip()
            ),
            "raw_path": str(self.config.get("HDFS_RAW_PATH", "")),
            "ads_path": str(self.config.get("HDFS_RESULT_ROOT", "")),
            "hadoop_cli": hadoop_cli,
            "hdfs_cli": hdfs_cli,
            "cli_detected": bool(hadoop_cli and hdfs_cli),
            "readiness_scope": "client_configuration_only",
        }

    def _spark_status(self, data_mode: str) -> dict:
        configured_binary = str(self.config.get("SPARK_SUBMIT_BIN", "spark-submit"))
        binary_path = self._resolve_executable(configured_binary)
        script = Path(str(self.config.get("SPARK_ANALYSIS_SCRIPT", "")))
        script_ready = script.is_file()
        return {
            "enabled": data_mode == "hdfs",
            "submit_binary": binary_path or configured_binary,
            "submit_executable": binary_path is not None,
            "analysis_script": str(script),
            "analysis_script_exists": script_ready,
            "score_engine": str(self.config.get("SPARK_SCORE_ENGINE", "pandas")),
            "ready": binary_path is not None and script_ready,
        }

    @staticmethod
    def _resolve_executable(value: str) -> str | None:
        path = Path(value)
        if path.is_absolute() or path.parent != Path("."):
            return str(path) if path.is_file() else None
        return shutil.which(value)

    def _model_status(self) -> dict:
        registry = ModelRegistry(
            self.config.get("MODEL_OUTPUT_DIR", "data/models"),
            self.config.get("MODEL_MANIFEST_PATH", "data/models/active_models.json"),
            self.config.get("CARDIO_FEATURE_COLUMNS", []),
        )
        try:
            paths = registry.load_active_models()
        except FileNotFoundError as exc:
            return {
                "ready": False,
                "heart_ready": False,
                "stroke_ready": False,
                "message": str(exc),
            }
        return {
            "ready": True,
            "heart_ready": bool(paths.get("heart")),
            "stroke_ready": bool(paths.get("stroke")),
            "message": "双风险模型已就绪。",
        }
