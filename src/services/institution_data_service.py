"""机构端当前数据集服务：统一管理看板的当前数据来源。"""

from __future__ import annotations

from src.services.analysis_task_service import AnalysisTaskService
from src.services.task_service import TaskService


class InstitutionDataService:
    """机构页面只读取已上传且分析成功的数据集，不读取训练基线文件。"""

    def __init__(self, config):
        self.config = config
        self.task_service = TaskService(config)
        self.analysis_service = AnalysisTaskService(config, task_service=self.task_service)

    def latest_successful_result(self):
        for dataset in self.task_service.database_dao.list_datasets(limit=100):
            if not dataset.get("result_path"):
                continue
            if dataset.get("validation_status") not in {"success", "completed", None}:
                continue
            try:
                result = self.analysis_service.read_result(dataset["dataset_id"])
            except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError):
                continue
            return dataset, result
        return None, None

    def dashboard(self):
        dataset, result = self.latest_successful_result()
        if not dataset or not result:
            return {
                "initialized": False,
                "has_data": False,
                "message": "当前尚未上传并完成居民数据分析，请先上传社区数据。",
                "data_source": None,
                "dataset_id": None,
                "data_period": None,
                "total_residents": 0,
                "districts": [],
                "region_analysis": [],
                "age_groups": [],
                "risk_factors": [],
                "follow_ups": [],
                "coverage": {
                    "coverage_level": "unknown",
                    "coverage_name": "等待上传数据",
                    "map_name": None,
                    "map_available": False,
                },
                "coverage_level": "unknown",
                "coverage_name": "等待上传数据",
                "map_name": None,
                "map_available": False,
                "risk_metric_source": "unavailable",
                "heart_risk_rate": None,
                "stroke_risk_rate": None,
                "comorbidity_rate": None,
                "high_risk_count": 0,
                "high_risk_follow_up_count": 0,
            }

        payload = dict(result)
        payload.update(
            {
                "initialized": True,
                "has_data": True,
                "data_source": dataset.get("filename"),
                "dataset_id": dataset.get("dataset_id"),
                "data_period": dataset.get("data_period") or result.get("data_period"),
                "message": "当前展示最近一次已完成分析的数据集。",
            }
        )
        payload.setdefault("region_analysis", payload.get("districts", []))
        payload.setdefault("districts", payload.get("region_analysis", []))
        payload.setdefault("follow_ups", [])
        payload.setdefault("high_risk_count", payload.get("high_risk_follow_up_count", 0))
        return payload

    def follow_ups(self, limit=100):
        data = self.dashboard()
        if not data.get("has_data"):
            return []
        return list(data.get("follow_ups", []))[:limit]
