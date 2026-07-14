"""阶段二任务服务：管理上传任务状态。"""

from __future__ import annotations

import re
import uuid

from src.dao.database_dao import DatabaseDAO, TASK_STATUSES
from src.dao.database_factory import create_database_dao


class TaskService:
    def __init__(self, config, database_dao: DatabaseDAO | None = None):
        self.config = config
        self.database_dao = database_dao or create_database_dao(config)

    def create_task(self, filename: str, file_size: int, total_chunks: int, storage_mode: str, **metadata):
        task_id = self._build_task_id(filename)
        task = self.database_dao.create_task(
            {
                "task_id": task_id,
                "filename": filename,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "status": "queued",
                "storage_mode": storage_mode,
                **metadata,
            }
        )
        return self._with_engine_metadata(task)

    def get_task(self, task_id: str):
        task = self.database_dao.get_task(task_id)
        if not task:
            raise FileNotFoundError(f"Task not found: {task_id}")
        return self._with_engine_metadata(task)

    def update_status(self, task_id: str, status: str, error_message: str | None = None):
        if status not in TASK_STATUSES:
            raise ValueError(f"Invalid task status: {status}")
        fields = {"status": status}
        if status == "running":
            fields.update({"started_at": self.database_dao.now(), "stage": "analysis"})
        if status in {"success", "failed", "cancelled"}:
            fields["finished_at"] = self.database_dao.now()
        if error_message is not None:
            fields["error_message"] = error_message
        self.database_dao.update_task(task_id, **fields)
        return self.get_task(task_id)

    def mark_success_if_complete(self, task_id: str):
        task = self.get_task(task_id)
        if task["total_chunks"] and task["uploaded_chunks"] >= task["total_chunks"]:
            return self.update_status(task_id, "success")
        return self._with_engine_metadata(task)

    def _with_engine_metadata(self, task: dict) -> dict:
        """在 API 响应层补充执行拓扑，不污染任务表结构。"""
        result = dict(task)
        distributed = str(result.get("storage_mode", "local")).lower() == "hdfs"
        result["storage_engine"] = "hdfs" if distributed else "local_filesystem"
        result["compute_engine"] = "apache_spark" if distributed else "local_pandas"
        return result

    @staticmethod
    def _build_task_id(filename: str) -> str:
        # 文件名只保留为展示字段，任务 ID 永不携带用户输入路径。
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(filename))[:40].strip(".")
        return f"{uuid.uuid4().hex}{('-' + safe_name) if safe_name else ''}"
