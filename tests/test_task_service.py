from pathlib import Path

from config import BaseConfig
from src.dao.database_dao import DatabaseDAO
from src.services.task_service import TaskService
from src.services.analysis_task_service import AnalysisTaskService
from src.services.platform_capability_service import PlatformCapabilityService


def test_task_lifecycle_and_status_validation(tmp_path):
    class LocalConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")

    dao = DatabaseDAO(LocalConfig.DATABASE_PATH)
    service = TaskService(LocalConfig.as_dict(), database_dao=dao)

    task = service.create_task("demo.csv", 10, 2, "local")
    assert task["status"] == "queued"
    assert task["storage_engine"] == "local_filesystem"
    assert task["compute_engine"] == "local_pandas"
    updated = service.update_status(task["task_id"], "running")
    assert updated["status"] == "running"
    updated = service.update_status(task["task_id"], "cancelled")
    assert updated["status"] == "cancelled"

    try:
        service.update_status(task["task_id"], "invalid")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_missing_task_raises(tmp_path):
    class LocalConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")

    dao = DatabaseDAO(LocalConfig.DATABASE_PATH)
    service = TaskService(LocalConfig.as_dict(), database_dao=dao)

    try:
        service.get_task("missing")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_native_spark_command_uses_configured_paths(tmp_path):
    class LocalConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        SPARK_SUBMIT_BIN = "/opt/spark/bin/spark-submit"
        SPARK_ANALYSIS_SCRIPT = "/srv/cardiospark/scripts/run_phase2_analysis.py"
        MODEL_MANIFEST_PATH = "/srv/cardiospark/data/models/active_models.json"

    service = AnalysisTaskService(LocalConfig.as_dict())
    command = service._build_spark_command("task-1", "/input.csv", "/output/task-1")

    assert command == [
        "/opt/spark/bin/spark-submit",
        "/srv/cardiospark/scripts/run_phase2_analysis.py",
        "--input", "/input.csv",
        "--output", "/output/task-1",
        "--task-id", "task-1",
        "--model-manifest", "/srv/cardiospark/data/models/active_models.json",
        "--score-engine", "pandas",
    ]

    assert service._qualify_hdfs_path("/user/demo/input.csv") == (
        "hdfs://localhost:9000/user/demo/input.csv"
    )
    assert service._qualify_hdfs_path("hdfs://namenode:9000/user/demo/input.csv") == (
        "hdfs://namenode:9000/user/demo/input.csv"
    )


def test_spark_command_resolves_bare_binary_when_available(tmp_path, monkeypatch):
    class LocalConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        SPARK_SUBMIT_BIN = "spark-submit"

    monkeypatch.setattr(
        "src.services.analysis_task_service.shutil.which",
        lambda value: "/opt/spark/bin/spark-submit.cmd" if value == "spark-submit" else None,
    )
    service = AnalysisTaskService(LocalConfig.as_dict())

    assert service._build_spark_command("task-1", "/input.csv", "/output")[0] == (
        "/opt/spark/bin/spark-submit.cmd"
    )


def test_distributed_capabilities_require_hdfs_client_and_spark_files(tmp_path, monkeypatch):
    spark_submit = tmp_path / "spark-submit.cmd"
    spark_submit.write_text("test", encoding="utf-8")
    analysis_script = tmp_path / "analysis.py"
    analysis_script.write_text("print('test')", encoding="utf-8")

    class FakeHDFS:
        @staticmethod
        def is_available():
            return True

    class HDFSConfig(BaseConfig):
        DATA_MODE = "hdfs"
        DATABASE_PATH = str(tmp_path / "app.db")
        SPARK_SUBMIT_BIN = str(spark_submit)
        SPARK_ANALYSIS_SCRIPT = str(analysis_script)
        MODEL_OUTPUT_DIR = str(tmp_path / "models")
        MODEL_MANIFEST_PATH = str(tmp_path / "models" / "missing.json")

    monkeypatch.setattr(
        "src.services.platform_capability_service.HDFSDao.from_config",
        lambda config: FakeHDFS(),
    )

    data = PlatformCapabilityService(HDFSConfig.as_dict()).inspect()

    assert data["hdfs"]["client_ready"] is True
    assert data["spark"]["submit_executable"] is True
    assert data["spark"]["analysis_script_exists"] is True
    assert data["compute_engine"] == "apache_spark"
    assert data["distributed_infrastructure_ready"] is True
    assert data["distributed_pipeline_ready"] is False
