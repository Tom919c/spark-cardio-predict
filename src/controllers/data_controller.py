from flask import Blueprint, current_app

from src.services.data_service import DataService
from src.utils.response import error_response, success_response

data_bp = Blueprint("data", __name__)


@data_bp.route("/profile")
def dataset_profile():
    service = DataService(current_app.config)
    profile = service.get_dataset_profile()
    if not profile["configured"]:
        return error_response(
            message="数据集路径未配置，请在 config/base.py 中填写 DATASET_FILE_PATH。",
            code=400,
            data=profile,
        )
    return success_response(message="数据集配置读取成功。", data=profile)


@data_bp.route("/preview")
def dataset_preview():
    service = DataService(current_app.config)
    preview = service.preview_dataset()
    if not preview["configured"]:
        return error_response(
            message="数据集路径未配置，请先填写 DATASET_FILE_PATH。",
            code=400,
            data=preview,
        )
    if not preview["exists"]:
        return error_response(
            message="数据集文件不存在，请检查 DATASET_FILE_PATH。",
            code=404,
            data=preview,
        )
    return success_response(message="数据集预览成功。", data=preview)


@data_bp.route("/preprocess")
def dataset_preprocess():
    service = DataService(current_app.config)
    processed = service.preprocess_dataset()
    if not processed["configured"]:
        return error_response(
            message="数据集路径未配置，请检查 DATASET_FILE_PATH。",
            code=400,
            data=processed,
        )
    if not processed["exists"]:
        return error_response(
            message="数据集文件不存在，请检查项目内数据集是否已放置。",
            code=404,
            data=processed,
        )
    if not processed["valid"]:
        return error_response(
            message="数据集字段校验未通过，请根据缺失字段补齐数据。",
            code=400,
            data=processed,
        )
    return success_response(message="数据预处理完成。", data=processed)
