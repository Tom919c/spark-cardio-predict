"""任务状态接口。"""

from __future__ import annotations

from flask import Blueprint, current_app

from src.services.task_service import TaskService
from src.utils.response import error_response, success_response


task_bp = Blueprint("task", __name__)


@task_bp.route("/tasks/<task_id>")
def get_task(task_id: str):
    service = TaskService(current_app.config)
    try:
        task = service.get_task(task_id)
    except FileNotFoundError as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="任务获取成功。", data=task)


@task_bp.route("/status/<task_id>")
def get_task_status(task_id: str):
    """阶段二前端轮询的稳定路径。"""
    return get_task(task_id)
