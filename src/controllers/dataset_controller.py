"""阶段二数据集预览和分析结果接口。"""

from flask import Blueprint, current_app

from src.services.analysis_task_service import AnalysisTaskService
from src.services.task_service import TaskService
from src.utils.field_mapping import canonicalize_pandas_columns
from src.utils.privacy import mask_dataframe
from src.utils.response import error_response, success_response


dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/<dataset_id>/preview")
def dataset_preview(dataset_id):
    try:
        dataset = TaskService(current_app.config).database_dao.get_dataset(dataset_id)
        if not dataset:
            raise FileNotFoundError("数据集不存在。")
        path = dataset.get("local_path")
        if not path:
            return success_response(
                message="HDFS 数据集需要通过分析任务读取预览。",
                data={"dataset": dataset, "preview_rows": []},
            )
        import pandas as pd

        frame, field_mapping = canonicalize_pandas_columns(pd.read_csv(path, nrows=20))
        frame = mask_dataframe(frame)
        return success_response(
            message="数据集预览成功。",
            data={
                "dataset": dataset,
                "columns": frame.columns.tolist(),
                "field_mapping": field_mapping,
                "preview_rows": frame.fillna("").to_dict(orient="records"),
            },
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return error_response(message=str(exc), code=404)


@dataset_bp.route("/<dataset_id>/result")
def dataset_result(dataset_id):
    try:
        result = AnalysisTaskService(current_app.config).read_result(dataset_id)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="数据集分析结果获取成功。", data=result)
