"""阶段二快照趋势接口。"""

from flask import Blueprint, current_app, request

from src.services.analysis_task_service import AnalysisTaskService
from src.services.snapshot_service import SnapshotService
from src.utils.response import error_response, success_response


trend_bp = Blueprint("trend", __name__)


@trend_bp.route("/snapshots")
def list_snapshots():
    service = AnalysisTaskService(current_app.config)
    snapshots = []
    for dataset in service.task_service.database_dao.list_datasets(limit=100):
        if not dataset.get("result_path"):
            continue
        try:
            result = service.read_result(dataset["dataset_id"])
        except (FileNotFoundError, OSError, ValueError, RuntimeError):
            continue
        snapshots.append(result)
    return success_response(
        message="历史快照读取成功。",
        data={"snapshots": snapshots, "count": len(snapshots)},
    )


@trend_bp.route("/compare", methods=["POST"])
def compare_snapshots():
    payload = request.get_json(silent=True) or {}
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        return error_response(message="snapshots 必须为数组。", code=400)
    result = SnapshotService().compare(snapshots)
    if not result["available"]:
        return error_response(message=result["message"], code=400, data=result)
    return success_response(message=result["message"], data=result)
