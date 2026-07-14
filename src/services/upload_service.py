"""阶段二上传服务：支持本地和可选 HDFS 分片上传。"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from src.dao.hdfs_dao import HDFSDao
from src.services.task_service import TaskService


class UploadService:
    def __init__(self, config, task_service: TaskService | None = None, hdfs_dao: HDFSDao | None = None):
        self.config = config
        self.upload_root = Path(config["UPLOAD_ROOT"])
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.task_service = task_service or TaskService(config)
        self.max_file_size = int(config.get("UPLOAD_MAX_FILE_SIZE", 512 * 1024 * 1024))
        self.hdfs_upload_root = config.get("HDFS_UPLOAD_ROOT", "")
        self.hdfs_raw_path = config.get("HDFS_RAW_PATH", "")
        self.hdfs_dao = hdfs_dao or HDFSDao.from_config(config)

    def create_upload_task(self, filename: str, file_size: int, total_chunks: int, use_hdfs: bool = False, data_period: str | None = None):
        original_name = Path(str(filename)).name
        if Path(original_name).suffix.lower() != ".csv":
            raise ValueError("目前只支持 CSV 文件。")
        safe_name = secure_filename(original_name) or f"upload-{uuid.uuid4().hex}.csv"
        if not safe_name.lower().endswith(".csv"):
            safe_name = f"{safe_name}.csv"
        if file_size <= 0 or file_size > self.max_file_size:
            raise ValueError("文件大小不符合配置限制。")
        if total_chunks <= 0:
            raise ValueError("total_chunks 必须为正整数。")
        data_period = str(data_period or "").strip()
        if not data_period or not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2]|Q[1-4])", data_period):
            raise ValueError("data_period 必须使用 YYYY-MM 或 YYYY-Qn 格式。")
        storage_mode = "hdfs" if use_hdfs else "local"
        if storage_mode == "hdfs" and not self.hdfs_dao.is_available():
            raise ValueError("当前未配置可用的 HDFS 客户端。")
        return self.task_service.create_task(
            safe_name,
            file_size,
            total_chunks,
            storage_mode,
            data_period=data_period,
        )

    def upload_chunk(self, task_id: str, chunk_index: int, data: bytes):
        task = self.task_service.get_task(task_id)
        chunk_index = int(chunk_index)
        if self.task_service.database_dao.has_chunk(task_id, chunk_index):
            return {"task": task, "duplicate": True, "completed": task["uploaded_chunks"] >= task["total_chunks"]}

        if chunk_index < 0 or chunk_index >= task["total_chunks"]:
            raise ValueError("chunk_index 超出范围。")
        if task["uploaded_chunks"] >= task["total_chunks"] and not self.is_task_complete(task_id):
            raise ValueError("上传任务状态异常，请重新创建任务。")
        if task["storage_mode"] == "hdfs":
            chunk_path = f"{self.hdfs_upload_root.rstrip('/')}/{task_id}/{chunk_index:06d}.part"
            self.hdfs_dao.ensure_dir(f"{self.hdfs_upload_root.rstrip('/')}/{task_id}")
            self.hdfs_dao.write_bytes(chunk_path, data, overwrite=True)
        else:
            task_dir = self.upload_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = task_dir / f"{chunk_index:06d}.part"
            chunk_path.write_bytes(data)
            chunk_path = str(chunk_path)
        self.task_service.database_dao.add_chunk(task_id, chunk_index, len(data), str(chunk_path))

        task = self.task_service.get_task(task_id)
        completed = self.is_task_complete(task_id)
        if completed:
            task = self.task_service.update_status(task_id, "success")
        else:
            task = self.task_service.update_status(task_id, "running")
        return {"task": task, "duplicate": False, "completed": completed}

    def is_task_complete(self, task_id: str) -> bool:
        task = self.task_service.get_task(task_id)
        if task["total_chunks"] <= 0:
            return False
        chunk_count = self.task_service.database_dao.count_chunks(task_id)
        if chunk_count != task["total_chunks"]:
            return False
        expected = list(range(task["total_chunks"]))
        actual = [item["chunk_index"] for item in self.task_service.database_dao.list_chunks(task_id)]
        return actual == expected

    def finalize_task(self, task_id: str):
        task = self.task_service.get_task(task_id)
        if task.get("dataset_id") and task.get("storage_path"):
            return task
        if not self.is_task_complete(task_id):
            raise ValueError("Upload chunks are incomplete or out of sequence.")
        chunks = list(self.task_service.database_dao.all_chunks(task_id))
        if task["storage_mode"] == "hdfs":
            target_path = f"{self.hdfs_raw_path.rstrip('/')}/{task_id}/{secure_filename(task['filename'])}"
            self.hdfs_dao.ensure_dir(f"{self.hdfs_raw_path.rstrip('/')}/{task_id}")
            self.hdfs_dao.merge_files([item["chunk_path"] for item in chunks], target_path)
            remote_status = self.hdfs_dao.status(target_path)
            if not remote_status or int(remote_status.get("length", -1)) != int(task["file_size"]):
                self.hdfs_dao.delete(target_path)
                raise ValueError("HDFS 合并文件大小与上传声明不一致。")
            storage_path = target_path
            local_path = None
        else:
            task_dir = self.upload_root / task_id
            target = task_dir / secure_filename(task["filename"])
            with target.open("wb") as output:
                for item in chunks:
                    with Path(item["chunk_path"]).open("rb") as chunk_file:
                        while True:
                            block = chunk_file.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
            if target.stat().st_size != int(task["file_size"]):
                raise ValueError("合并文件大小与上传声明不一致。")
            storage_path = str(target)
            local_path = str(target)
        file_sha256 = self._sha256_local(Path(local_path)) if local_path else None
        dataset_id = f"ds_{uuid.uuid4().hex}"
        self.task_service.database_dao.create_dataset(
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "filename": task["filename"],
                "file_size": task["file_size"],
                "file_sha256": file_sha256,
                "storage_mode": task["storage_mode"],
                "hdfs_path": storage_path if task["storage_mode"] == "hdfs" else None,
                "local_path": local_path,
                "data_period": task.get("data_period"),
                "validation_status": "pending",
            }
        )
        self.task_service.database_dao.update_task(
            task_id,
            dataset_id=dataset_id,
            storage_path=storage_path,
            file_sha256=file_sha256,
            progress=100,
            stage="uploaded",
        )
        self.task_service.update_status(task_id, "success")
        return self.task_service.get_task(task_id)

    @staticmethod
    def _sha256_local(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
