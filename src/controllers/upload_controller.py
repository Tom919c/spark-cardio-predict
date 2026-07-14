"""上传接口。"""

from __future__ import annotations

from flask import Blueprint, current_app, request

from src.services.upload_service import UploadService
from src.services.analysis_task_service import AnalysisTaskService
from src.services.task_service import TaskService
from src.utils.response import error_response, success_response


upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/tasks", methods=["POST"])
def create_task():
    payload = request.get_json(silent=True) or {}
    service = UploadService(current_app.config)
    try:
        task = service.create_upload_task(
            filename=payload["filename"],
            file_size=int(payload.get("file_size", 0)),
            total_chunks=int(payload.get("total_chunks", 0)),
            data_period=payload.get("data_period"),
        )
    except KeyError as exc:
        return error_response(message=f"Missing field: {exc.args[0]}", code=400)
    except ValueError as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="任务创建成功。", data=task)


@upload_bp.route("/init", methods=["POST"])
def init_upload():
    """文档约定的分片上传初始化别名。"""
    return create_task()


@upload_bp.route("/<task_id>", methods=["GET"])
def upload_status(task_id: str):
    try:
        task = TaskService(current_app.config).get_task(task_id)
    except FileNotFoundError as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="上传任务状态获取成功。", data=task)


@upload_bp.route("/tasks/<task_id>/chunks", methods=["POST"])
def upload_chunk(task_id: str):
    service = UploadService(current_app.config)
    if "chunk" not in request.files:
        return error_response(message="Missing file chunk.", code=400)
    chunk_file = request.files["chunk"]
    chunk_index = request.form.get("chunk_index", type=int)
    if chunk_index is None:
        return error_response(message="Missing chunk_index.", code=400)
    try:
        result = service.upload_chunk(task_id, chunk_index, chunk_file.read())
    except (FileNotFoundError, ValueError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="分片上传成功。", data=result)


@upload_bp.route("/<task_id>/chunks", methods=["PUT", "POST"])
def upload_chunk_alias(task_id: str):
    if "chunk" in request.files:
        return upload_chunk(task_id)
    chunk_index = request.args.get("chunk_index", request.headers.get("X-Chunk-Index"), type=int)
    if chunk_index is None:
        return error_response(message="缺少 chunk_index。", code=400)
    try:
        result = UploadService(current_app.config).upload_chunk(
            task_id, chunk_index, request.get_data()
        )
    except (FileNotFoundError, ValueError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="分片上传成功。", data=result)


@upload_bp.route("/tasks/<task_id>/finalize", methods=["POST"])
def finalize_task(task_id: str):
    service = UploadService(current_app.config)
    try:
        task = service.finalize_task(task_id)
        if task.get("dataset_id"):
            task = AnalysisTaskService(current_app.config).submit(
                task_id, task["dataset_id"]
            )
    except (FileNotFoundError, ValueError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="任务已完成。", data=task)


@upload_bp.route("/<task_id>/complete", methods=["POST"])
def complete_upload(task_id: str):
    return finalize_task(task_id)
