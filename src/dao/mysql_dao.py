"""MySQL 任务/数据集 DAO，按需加载 PyMySQL。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


class MySQLDatabaseDAO:
    """与 SQLite DatabaseDAO 保持相同方法契约的 MySQL 实现。"""

    def __init__(self, config):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("DATABASE_TYPE=mysql 需要安装 PyMySQL。") from exc
        self.pymysql = pymysql
        self.config = config
        self._init_schema()

    @contextmanager
    def _connect(self):
        connection = self.pymysql.connect(
            host=self.config.get("MYSQL_HOST", "127.0.0.1"),
            port=int(self.config.get("MYSQL_PORT", 3306)),
            user=self.config.get("MYSQL_USER", "root"),
            password=self.config.get("MYSQL_PASSWORD", ""),
            database=self.config.get("MYSQL_DATABASE", "cardiospark"),
            charset="utf8mb4",
            cursorclass=self.pymysql.cursors.DictCursor,
            autocommit=False,
        )
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def _init_schema(self):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upload_tasks (
                        task_id VARCHAR(128) PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        file_size BIGINT NOT NULL DEFAULT 0,
                        total_chunks INT NOT NULL DEFAULT 0,
                        uploaded_chunks INT NOT NULL DEFAULT 0,
                        status VARCHAR(20) NOT NULL,
                        storage_mode VARCHAR(20) NOT NULL DEFAULT 'local',
                        storage_path TEXT,
                        dataset_id VARCHAR(128),
                        data_period VARCHAR(32),
                        result_path TEXT,
                        stage VARCHAR(32),
                        progress DOUBLE NOT NULL DEFAULT 0,
                        started_at VARCHAR(64),
                        finished_at VARCHAR(64),
                        file_sha256 VARCHAR(128),
                        error_message TEXT,
                        created_at VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upload_chunks (
                        task_id VARCHAR(128) NOT NULL,
                        chunk_index INT NOT NULL,
                        chunk_size BIGINT NOT NULL,
                        chunk_path TEXT NOT NULL,
                        created_at VARCHAR(64) NOT NULL,
                        PRIMARY KEY (task_id, chunk_index)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        dataset_id VARCHAR(128) PRIMARY KEY,
                        task_id VARCHAR(128) NOT NULL,
                        filename VARCHAR(255) NOT NULL,
                        file_size BIGINT NOT NULL DEFAULT 0,
                        file_sha256 VARCHAR(128),
                        storage_mode VARCHAR(20) NOT NULL DEFAULT 'local',
                        hdfs_path TEXT,
                        local_path TEXT,
                        data_period VARCHAR(32),
                        region_level VARCHAR(32),
                        region_name VARCHAR(255),
                        row_count BIGINT,
                        validation_status VARCHAR(32) NOT NULL DEFAULT 'pending',
                        result_path TEXT,
                        upload_time VARCHAR(64) NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assessment_records (
                        assessment_id VARCHAR(128) PRIMARY KEY,
                        client_hash VARCHAR(128) NOT NULL,
                        model_version VARCHAR(128),
                        knowledge_version VARCHAR(128),
                        heart_probability DOUBLE NOT NULL,
                        stroke_probability DOUBLE NOT NULL,
                        risk_level_code INT NOT NULL,
                        risk_level_name VARCHAR(64) NOT NULL,
                        final_category INT NOT NULL,
                        input_json LONGTEXT NOT NULL,
                        result_json LONGTEXT NOT NULL,
                        created_at VARCHAR(64) NOT NULL,
                        INDEX idx_assessment_client_created (client_hash, created_at)
                    )
                    """
                )

    def create_task(self, task: dict[str, Any]):
        now = self.now()
        payload = {
            "task_id": task["task_id"], "filename": task["filename"],
            "file_size": int(task.get("file_size", 0)),
            "total_chunks": int(task.get("total_chunks", 0)),
            "uploaded_chunks": int(task.get("uploaded_chunks", 0)),
            "status": task.get("status", "queued"),
            "storage_mode": task.get("storage_mode", "local"),
            "storage_path": task.get("storage_path"),
            "dataset_id": task.get("dataset_id"), "data_period": task.get("data_period"),
            "result_path": task.get("result_path"), "stage": task.get("stage", "upload"),
            "progress": float(task.get("progress", 0)), "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"), "file_sha256": task.get("file_sha256"),
            "error_message": task.get("error_message"), "created_at": now, "updated_at": now,
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(["%s"] * len(payload))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO upload_tasks ({columns}) VALUES ({placeholders})",
                    tuple(payload.values()),
                )
        return payload

    def get_task(self, task_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM upload_tasks WHERE task_id=%s", (task_id,))
                return cursor.fetchone()

    def list_chunks(self, task_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT chunk_index,chunk_size,chunk_path,created_at FROM upload_chunks WHERE task_id=%s ORDER BY chunk_index", (task_id,))
                return list(cursor.fetchall())

    def has_chunk(self, task_id, chunk_index):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM upload_chunks WHERE task_id=%s AND chunk_index=%s", (task_id, chunk_index))
                return cursor.fetchone() is not None

    def add_chunk(self, task_id, chunk_index, chunk_size, chunk_path):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT IGNORE INTO upload_chunks(task_id,chunk_index,chunk_size,chunk_path,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (task_id, int(chunk_index), int(chunk_size), chunk_path, self.now()),
                )
                cursor.execute("SELECT COUNT(1) AS count FROM upload_chunks WHERE task_id=%s", (task_id,))
                count = cursor.fetchone()["count"]
                cursor.execute("UPDATE upload_tasks SET uploaded_chunks=%s,updated_at=%s WHERE task_id=%s", (count, self.now(), task_id))

    def update_task(self, task_id, **fields):
        allowed = {"filename", "file_size", "total_chunks", "uploaded_chunks", "status", "storage_mode", "storage_path", "dataset_id", "data_period", "result_path", "stage", "progress", "started_at", "finished_at", "file_sha256", "error_message"}
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"不允许更新任务字段: {sorted(invalid)}")
        if not fields:
            return
        fields["updated_at"] = self.now()
        columns = ", ".join(f"{key}=%s" for key in fields)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE upload_tasks SET {columns} WHERE task_id=%s", tuple(fields.values()) + (task_id,))

    def count_chunks(self, task_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(1) AS count FROM upload_chunks WHERE task_id=%s", (task_id,))
                return int(cursor.fetchone()["count"])

    def delete_task(self, task_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM upload_chunks WHERE task_id=%s", (task_id,))
                cursor.execute("DELETE FROM upload_tasks WHERE task_id=%s", (task_id,))

    def all_chunks(self, task_id):
        return self.list_chunks(task_id)

    def create_dataset(self, dataset):
        payload = {
            "dataset_id": dataset["dataset_id"], "task_id": dataset["task_id"], "filename": dataset["filename"],
            "file_size": int(dataset.get("file_size", 0)), "file_sha256": dataset.get("file_sha256"),
            "storage_mode": dataset.get("storage_mode", "local"), "hdfs_path": dataset.get("hdfs_path"),
            "local_path": dataset.get("local_path"), "data_period": dataset.get("data_period"),
            "region_level": dataset.get("region_level"), "region_name": dataset.get("region_name"),
            "row_count": dataset.get("row_count"), "validation_status": dataset.get("validation_status", "pending"),
            "result_path": dataset.get("result_path"), "upload_time": self.now(),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(["%s"] * len(payload))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"INSERT INTO datasets ({columns}) VALUES ({placeholders})", tuple(payload.values()))
        return payload

    def get_dataset(self, dataset_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM datasets WHERE dataset_id=%s", (dataset_id,))
                return cursor.fetchone()

    def list_datasets(self, limit=50):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM datasets ORDER BY upload_time DESC LIMIT %s", (int(limit),))
                return list(cursor.fetchall())

    def update_dataset(self, dataset_id, **fields):
        allowed = {"file_sha256", "hdfs_path", "local_path", "data_period", "region_level", "region_name", "row_count", "validation_status", "result_path"}
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"不允许更新数据集字段: {sorted(invalid)}")
        if not fields:
            return
        columns = ", ".join(f"{key}=%s" for key in fields)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE datasets SET {columns} WHERE dataset_id=%s", tuple(fields.values()) + (dataset_id,))

    def create_assessment(self, assessment):
        payload = {
            "assessment_id": assessment["assessment_id"],
            "client_hash": assessment["client_hash"],
            "model_version": assessment.get("model_version"),
            "knowledge_version": assessment.get("knowledge_version"),
            "heart_probability": float(assessment.get("heart_probability", 0)),
            "stroke_probability": float(assessment.get("stroke_probability", 0)),
            "risk_level_code": int(assessment.get("risk_level_code", 1)),
            "risk_level_name": assessment.get("risk_level_name", "I级：健康"),
            "final_category": int(assessment.get("final_category", 0)),
            "input_json": assessment.get("input_json", "{}"),
            "result_json": assessment.get("result_json", "{}"),
            "created_at": assessment.get("created_at") or self.now(),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(["%s"] * len(payload))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO assessment_records ({columns}) VALUES ({placeholders})",
                    tuple(payload.values()),
                )
        return payload

    def get_assessment(self, assessment_id):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM assessment_records WHERE assessment_id=%s",
                    (assessment_id,),
                )
                return cursor.fetchone()

    def list_assessments(self, client_hash, limit=10):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM assessment_records
                    WHERE client_hash=%s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (client_hash, int(limit)),
                )
                return list(cursor.fetchall())
