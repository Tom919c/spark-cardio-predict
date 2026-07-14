"""SQLite 默认的任务与分片元数据访问层。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_STATUSES = ("queued", "running", "success", "failed", "cancelled")


class DatabaseDAO:
    """轻量 SQLite DAO，使用参数化 SQL。"""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS upload_tasks (
                    task_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    uploaded_chunks INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    storage_mode TEXT NOT NULL DEFAULT 'local',
                    storage_path TEXT,
                    dataset_id TEXT,
                    data_period TEXT,
                    result_path TEXT,
                    stage TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    file_sha256 TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS upload_chunks (
                    task_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, chunk_index),
                    FOREIGN KEY (task_id) REFERENCES upload_tasks(task_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_sha256 TEXT,
                    storage_mode TEXT NOT NULL DEFAULT 'local',
                    hdfs_path TEXT,
                    local_path TEXT,
                    data_period TEXT,
                    region_level TEXT,
                    region_name TEXT,
                    row_count INTEGER,
                    validation_status TEXT NOT NULL DEFAULT 'pending',
                    result_path TEXT,
                    upload_time TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES upload_tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS assessment_records (
                    assessment_id TEXT PRIMARY KEY,
                    client_hash TEXT NOT NULL,
                    model_version TEXT,
                    knowledge_version TEXT,
                    heart_probability REAL NOT NULL,
                    stroke_probability REAL NOT NULL,
                    risk_level_code INTEGER NOT NULL,
                    risk_level_name TEXT NOT NULL,
                    final_category INTEGER NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assessment_client_created
                ON assessment_records(client_hash, created_at DESC);
                """
            )
            self._ensure_task_columns(connection)

    @staticmethod
    def _ensure_task_columns(connection):
        existing = {
            row[1]
            for row in connection.execute("PRAGMA table_info(upload_tasks)").fetchall()
        }
        additions = {
            "dataset_id": "TEXT",
            "data_period": "TEXT",
            "result_path": "TEXT",
            "stage": "TEXT",
            "progress": "REAL NOT NULL DEFAULT 0",
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "file_sha256": "TEXT",
        }
        for name, definition in additions.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE upload_tasks ADD COLUMN {name} {definition}")

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_task(self, task: dict[str, Any]) -> dict[str, Any]:
        now = self.now()
        payload = {
            "task_id": task["task_id"],
            "filename": task["filename"],
            "file_size": int(task.get("file_size", 0)),
            "total_chunks": int(task.get("total_chunks", 0)),
            "uploaded_chunks": int(task.get("uploaded_chunks", 0)),
            "status": task.get("status", "queued"),
            "storage_mode": task.get("storage_mode", "local"),
            "storage_path": task.get("storage_path"),
            "dataset_id": task.get("dataset_id"),
            "data_period": task.get("data_period"),
            "result_path": task.get("result_path"),
            "stage": task.get("stage", "upload"),
            "progress": float(task.get("progress", 0)),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "file_sha256": task.get("file_sha256"),
            "error_message": task.get("error_message"),
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO upload_tasks (
                    task_id, filename, file_size, total_chunks, uploaded_chunks,
                    status, storage_mode, storage_path, dataset_id, data_period,
                    result_path, stage, progress, started_at, finished_at,
                    file_sha256, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(payload.values()),
            )
        return payload

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM upload_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_chunks(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_index, chunk_size, chunk_path, created_at
                FROM upload_chunks
                WHERE task_id = ?
                ORDER BY chunk_index ASC
                """,
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_chunk(self, task_id: str, chunk_index: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM upload_chunks
                WHERE task_id = ? AND chunk_index = ?
                """,
                (task_id, chunk_index),
            ).fetchone()
        return row is not None

    def add_chunk(self, task_id: str, chunk_index: int, chunk_size: int, chunk_path: str):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO upload_chunks (
                    task_id, chunk_index, chunk_size, chunk_path, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, int(chunk_index), int(chunk_size), chunk_path, self.now()),
            )
            uploaded_chunks = connection.execute(
                "SELECT COUNT(1) AS count FROM upload_chunks WHERE task_id = ?",
                (task_id,),
            ).fetchone()["count"]
            connection.execute(
                """
                UPDATE upload_tasks
                SET uploaded_chunks = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (uploaded_chunks, self.now(), task_id),
            )

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "filename", "file_size", "total_chunks", "uploaded_chunks", "status",
            "storage_mode", "storage_path", "dataset_id", "data_period", "result_path",
            "stage", "progress", "started_at", "finished_at", "file_sha256", "error_message",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"不允许更新任务字段: {sorted(invalid)}")
        fields["updated_at"] = self.now()
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        values.append(task_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE upload_tasks SET {columns} WHERE task_id = ?",
                tuple(values),
            )

    def delete_task(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM upload_chunks WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM upload_tasks WHERE task_id = ?", (task_id,))

    def count_chunks(self, task_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(1) AS count FROM upload_chunks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row["count"])

    def all_chunks(self, task_id: str) -> Iterable[dict[str, Any]]:
        return self.list_chunks(task_id)

    def create_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        now = self.now()
        payload = {
            "dataset_id": dataset["dataset_id"],
            "task_id": dataset["task_id"],
            "filename": dataset["filename"],
            "file_size": int(dataset.get("file_size", 0)),
            "file_sha256": dataset.get("file_sha256"),
            "storage_mode": dataset.get("storage_mode", "local"),
            "hdfs_path": dataset.get("hdfs_path"),
            "local_path": dataset.get("local_path"),
            "data_period": dataset.get("data_period"),
            "region_level": dataset.get("region_level"),
            "region_name": dataset.get("region_name"),
            "row_count": dataset.get("row_count"),
            "validation_status": dataset.get("validation_status", "pending"),
            "result_path": dataset.get("result_path"),
            "upload_time": now,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id, task_id, filename, file_size, file_sha256,
                    storage_mode, hdfs_path, local_path, data_period,
                    region_level, region_name, row_count, validation_status,
                    result_path, upload_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(payload.values()),
            )
        return payload

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_datasets(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY upload_time DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_dataset(self, dataset_id: str, **fields: Any) -> None:
        allowed = {
            "file_sha256", "hdfs_path", "local_path", "data_period", "region_level",
            "region_name", "row_count", "validation_status", "result_path",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"不允许更新数据集字段: {sorted(invalid)}")
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [dataset_id]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE datasets SET {columns} WHERE dataset_id = ?", tuple(values)
            )

    def create_assessment(self, assessment: dict[str, Any]) -> dict[str, Any]:
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
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assessment_records (
                    assessment_id, client_hash, model_version, knowledge_version,
                    heart_probability, stroke_probability, risk_level_code,
                    risk_level_name, final_category, input_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(payload.values()),
            )
        return payload

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assessment_records WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_assessments(self, client_hash: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assessment_records
                WHERE client_hash = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (client_hash, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]
