"""上传完成后的异步分析任务编排。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from src.dao.hdfs_dao import HDFSDao
from src.services.phase2_analysis_service import Phase2AnalysisService
from src.services.task_service import TaskService


class AnalysisTaskService:
    """使用独立线程解耦上传请求和本地/Spark 分析。"""

    _executors = {}
    _lock = Lock()

    def __init__(self, config, task_service=None):
        self.config = config
        self.task_service = task_service or TaskService(config)
        key = str(config.get("DATABASE_PATH", "default"))
        with self._lock:
            if key not in self._executors:
                self._executors[key] = ThreadPoolExecutor(
                    max_workers=int(config.get("TASK_WORKERS", 2)),
                    thread_name_prefix="cardiospark-phase2",
                )
            self.executor = self._executors[key]

    def submit(self, task_id: str, dataset_id: str):
        current = self.task_service.get_task(task_id)
        if current.get("stage") in {"queued", "analysis", "completed"} and current.get("dataset_id") == dataset_id:
            return current
        self.task_service.database_dao.update_task(
            task_id, dataset_id=dataset_id, stage="queued", progress=0, finished_at=None
        )
        self.task_service.update_status(task_id, "queued")
        self.executor.submit(self._run, task_id, dataset_id)
        return self.task_service.get_task(task_id)

    def _run(self, task_id, dataset_id):
        try:
            self.task_service.update_status(task_id, "running")
            dataset = self.task_service.database_dao.get_dataset(dataset_id)
            if not dataset:
                raise FileNotFoundError("数据集元数据不存在。")
            if str(self.config.get("DATA_MODE", "local")).lower() == "hdfs":
                self._run_spark(task_id, dataset)
            else:
                self._run_local(task_id, dataset)
        except Exception as exc:  # 后台线程必须把错误持久化，不能让前端永久等待。
            self.task_service.database_dao.update_task(
                task_id,
                stage="failed",
                progress=100,
                error_message=str(exc)[:1000],
            )
            self.task_service.update_status(task_id, "failed", error_message=str(exc)[:1000])

    def _run_local(self, task_id, dataset):
        local_path = dataset.get("local_path")
        if not local_path:
            raise FileNotFoundError("本地模式缺少数据集文件路径。")
        result_dir = Path(self.config.get("LOCAL_RESULT_ROOT", "data/localstorage/results")) / dataset["dataset_id"]
        self.task_service.database_dao.update_task(task_id, stage="analysis", progress=10)
        result = Phase2AnalysisService(self.config).analyze_file(
            local_path,
            result_dir,
            metadata={
                "dataset_id": dataset["dataset_id"],
                "data_period": dataset.get("data_period"),
            },
        )
        self.task_service.database_dao.update_dataset(
            dataset["dataset_id"],
            region_level=result.get("coverage_level"),
            region_name=result.get("coverage_name"),
            row_count=result.get("total_residents"),
            validation_status="success",
            result_path=str(result_dir),
        )
        self.task_service.database_dao.update_task(
            task_id,
            stage="completed",
            progress=100,
            result_path=str(result_dir),
        )
        self.task_service.update_status(task_id, "success")

    def _run_spark(self, task_id, dataset):
        input_path = dataset.get("hdfs_path")
        if not input_path:
            raise FileNotFoundError("HDFS 模式缺少原始数据路径。")
        result_path = f"{self.config.get('HDFS_RESULT_ROOT', '').rstrip('/')}/{task_id}"
        command = self._build_spark_command(
            task_id,
            self._qualify_hdfs_path(input_path),
            self._qualify_hdfs_path(result_path),
        )
        environment = os.environ.copy()
        environment["PYSPARK_PYTHON"] = self.config.get("SPARK_PYTHON", sys.executable)
        environment["PYSPARK_DRIVER_PYTHON"] = environment["PYSPARK_PYTHON"]
        project_root = str(Path(__file__).resolve().parents[2])
        current_python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (project_root, current_python_path) if item
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=3600,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "Spark 作业失败")[-2000:])
        self.task_service.database_dao.update_dataset(
            dataset["dataset_id"],
            validation_status="success",
            result_path=result_path,
        )
        self.task_service.database_dao.update_task(
            task_id, stage="completed", progress=100, result_path=result_path
        )
        self.task_service.update_status(task_id, "success")

    def _qualify_hdfs_path(self, path: str) -> str:
        """将配置中的绝对 HDFS 路径绑定到当前环境的 NameNode。"""
        value = str(path or "").strip()
        if not value or "://" in value or not value.startswith("/"):
            return value
        namenode = str(self.config.get("HDFS_NAMENODE_URI", "")).strip().rstrip("/")
        return f"{namenode}{value}" if namenode else value

    def _build_spark_command(self, task_id, input_path, output_path):
        """构建当前环境中的原生 spark-submit 命令。"""
        script = str(
            self.config.get("SPARK_ANALYSIS_SCRIPT", "scripts/run_phase2_analysis.py")
        )
        spark_submit = str(self.config.get("SPARK_SUBMIT_BIN", "spark-submit"))
        # Windows commonly exposes Spark through spark-submit.cmd.  Resolving a
        # bare command here keeps subprocess.run independent of shell lookup
        # rules while preserving explicitly configured absolute paths.
        if not Path(spark_submit).is_absolute():
            spark_submit = shutil.which(spark_submit) or spark_submit
        return [
            spark_submit,
            script,
            "--input", input_path,
            "--output", output_path,
            "--task-id", task_id,
            "--model-manifest", str(self.config.get("MODEL_MANIFEST_PATH", "")),
            "--score-engine", str(self.config.get("SPARK_SCORE_ENGINE", "pandas")),
        ]

    def read_result(self, dataset_id):
        dataset = self.task_service.database_dao.get_dataset(dataset_id)
        if not dataset:
            raise FileNotFoundError("数据集不存在。")
        result_path = dataset.get("result_path")
        if not result_path:
            raise FileNotFoundError("数据集分析尚未完成。")
        if dataset.get("storage_mode") == "hdfs":
            client = HDFSDao.from_config(self.config)
            result_file = f"{result_path.rstrip('/')}/summary.json"
            if not client.exists(result_file):
                raise FileNotFoundError("HDFS 分析结果不存在。")
            return json.loads(client.read_text(result_file))
        result_file = Path(result_path) / "summary.json"
        if not result_file.exists():
            raise FileNotFoundError("分析结果文件不存在。")
        return json.loads(result_file.read_text(encoding="utf-8"))
