"""上传数据字段别名解析与映射记录。"""

from __future__ import annotations

from typing import Iterable


# 只在标准字段缺失时使用别名，避免静默覆盖用户已经提供的标准字段。
FIELD_ALIASES = {
    "age": ("年龄", "age_years"),
    "gender": ("性别", "sex"),
    "bmi": ("体重指数", "body_mass_index"),
    "cholesterol": ("胆固醇", "胆固醇情况", "cholesterol_level"),
    "diabetes": ("糖尿病", "糖尿病情况"),
    "hypertension": ("高血压", "高血压情况"),
    "smoker": ("吸烟", "吸烟情况", "smoking"),
    "alcohol": ("饮酒", "饮酒情况", "drinking"),
    "exercise": ("运动", "运动情况", "physical_activity"),
    "label_heart": ("心脏事件标签", "心脏标签", "heart_label"),
    "label_stroke": ("卒中事件标签", "脑卒中标签", "stroke_label"),
    "province": ("省",),
    "city": ("城市",),
    "region": ("地区",),
    "district": ("区县", "行政区"),
    "community": ("社区",),
    "community_id": ("社区编号", "社区ID"),
}


def resolve_field_mapping(columns: Iterable[str]) -> dict:
    """返回标准字段到实际来源字段的显式映射。"""
    available = {str(column).strip() for column in columns}
    mapping = {}
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical in available:
            mapping[canonical] = canonical
            continue
        source = next((alias for alias in aliases if alias in available), None)
        if source:
            mapping[canonical] = source
    return mapping


def canonicalize_pandas_columns(dataframe):
    """将可识别别名转换为标准字段，并返回映射记录。"""
    mapping = resolve_field_mapping(dataframe.columns)
    rename_map = {
        source: canonical
        for canonical, source in mapping.items()
        if source != canonical
    }
    return dataframe.rename(columns=rename_map), mapping
