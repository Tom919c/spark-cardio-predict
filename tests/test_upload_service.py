from pathlib import Path

from config import BaseConfig
from src.dao.database_dao import DatabaseDAO
from src.services.task_service import TaskService
from src.services.upload_service import UploadService


class TestConfig(BaseConfig):
    DATABASE_PATH = str(Path.cwd() / "tests_tmp_upload.db")
    UPLOAD_ROOT = str(Path.cwd() / "tests_uploads")


def test_chunk_upload_supports_out_of_order_and_duplicates(tmp_path, monkeypatch):
    class LocalConfig(TestConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")

    dao = DatabaseDAO(LocalConfig.DATABASE_PATH)
    task_service = TaskService(LocalConfig.as_dict(), database_dao=dao)
    upload_service = UploadService(LocalConfig.as_dict(), task_service=task_service)

    task = upload_service.create_upload_task("sample.csv", 12, 3, data_period="2026-07")
    task_id = task["task_id"]

    result2 = upload_service.upload_chunk(task_id, 1, b"bbb")
    result1 = upload_service.upload_chunk(task_id, 0, b"aaa")
    duplicate = upload_service.upload_chunk(task_id, 1, b"bbb")
    result3 = upload_service.upload_chunk(task_id, 2, b"ccc")

    assert result2["task"]["status"] == "running"
    assert result1["task"]["status"] == "running"
    assert duplicate["duplicate"] is True
    assert result3["completed"] is True
    assert upload_service.is_task_complete(task_id) is True
    assert task_service.get_task(task_id)["status"] == "success"


def test_finalize_rejects_incomplete_upload(tmp_path):
    class LocalConfig(TestConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")

    dao = DatabaseDAO(LocalConfig.DATABASE_PATH)
    task_service = TaskService(LocalConfig.as_dict(), database_dao=dao)
    upload_service = UploadService(LocalConfig.as_dict(), task_service=task_service)

    task = upload_service.create_upload_task("sample.csv", 12, 2, data_period="2026-07")
    upload_service.upload_chunk(task["task_id"], 0, b"aaa")

    try:
        upload_service.finalize_task(task["task_id"])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "incomplete" in str(exc).lower()


def test_upload_requires_snapshot_period(tmp_path):
    class LocalConfig(TestConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")

    service = UploadService(LocalConfig.as_dict())

    try:
        service.create_upload_task("sample.csv", 12, 2)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "data_period" in str(exc)


def test_server_data_mode_overrides_client_storage_preference(tmp_path):
    class FakeHDFS:
        @staticmethod
        def is_available():
            return True

    class HDFSConfig(TestConfig):
        DATA_MODE = "hdfs"
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")

    service = UploadService(HDFSConfig.as_dict(), hdfs_dao=FakeHDFS())

    task = service.create_upload_task(
        "sample.csv", 12, 2, use_hdfs=False, data_period="2026-07"
    )

    assert task["storage_mode"] == "hdfs"
    assert task["storage_engine"] == "hdfs"
    assert task["compute_engine"] == "apache_spark"


def test_local_mode_cannot_be_switched_to_hdfs_by_client(tmp_path):
    class LocalConfig(TestConfig):
        DATA_MODE = "local"
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")

    service = UploadService(LocalConfig.as_dict())
    task = service.create_upload_task(
        "sample.csv", 12, 2, use_hdfs=True, data_period="2026-07"
    )

    assert task["storage_mode"] == "local"
    assert task["storage_engine"] == "local_filesystem"
    assert task["compute_engine"] == "local_pandas"
