"""区域覆盖范围识别和群体指标聚合。"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


class RegionService:
    """把不同来源的区域字段转换为前端稳定的区域分析契约。"""

    def __init__(self, min_group_size: int = 5):
        self.min_group_size = int(min_group_size)

    def infer_coverage(self, dataframe: pd.DataFrame) -> dict:
        columns = set(dataframe.columns)
        province_values = self._values(dataframe, "province")
        city_values = self._values(dataframe, "city")
        region_values = self._values(dataframe, "region")
        district_values = self._values(dataframe, "district")
        community_values = self._values(dataframe, "community_id")
        if not community_values:
            community_values = self._values(dataframe, "community")

        parent_values = city_values or region_values
        if len(district_values) > 1:
            return {
                "coverage_level": "district",
                "coverage_name": self._first(parent_values) or "区域数据",
                "map_name": "chengdu" if self._is_chengdu(parent_values) else None,
                "map_available": self._is_chengdu(parent_values),
                "region_field": "district",
                "region_count": len(district_values),
            }
        if len(community_values) > 1:
            return {
                "coverage_level": "community",
                "coverage_name": self._first(district_values) or self._first(parent_values) or "社区数据",
                "map_name": None,
                "map_available": False,
                "region_field": "community_id" if "community_id" in columns else "community",
                "region_count": len(community_values),
            }
        if len(district_values) == 1:
            # 单个区县没有足够的区域层级可绘制，明确降级为表格，
            # 避免把整个成都市地图误用来表示一个区县的数据。
            return {
                "coverage_level": "table",
                "coverage_name": self._first(district_values),
                "map_name": None,
                "map_available": False,
                "region_field": "district",
                "region_count": 1,
            }
        if len(parent_values) > 1:
            return {
                "coverage_level": "city",
                "coverage_name": "多区域数据",
                "map_name": None,
                "map_available": False,
                "region_field": "city" if city_values else "region",
                "region_count": len(parent_values),
            }
        if province_values:
            return {
                "coverage_level": "province",
                "coverage_name": self._first(province_values),
                "map_name": None,
                "map_available": False,
                "region_field": "province",
                "region_count": len(province_values),
            }
        return {
            "coverage_level": "unknown",
            "coverage_name": "未识别区域",
            "map_name": None,
            "map_available": False,
            "region_field": None,
            "region_count": 0,
        }

    def aggregate(self, dataframe: pd.DataFrame, model_scores=None) -> dict:
        coverage = self.infer_coverage(dataframe)
        group_field = coverage.get("region_field")
        if not group_field or group_field not in dataframe.columns:
            group_field = None
        working = dataframe.copy()
        if group_field is None:
            working["_region"] = coverage["coverage_name"]
            group_field = "_region"
        working[group_field] = working[group_field].fillna("未知").astype(str)
        grouped = defaultdict(lambda: {
            "residents": 0,
            "heart_positive": 0,
            "stroke_positive": 0,
            "heart_high": 0,
            "stroke_high": 0,
            "comorbidity": 0,
        })
        heart_labels_available = "label_heart" in working.columns
        stroke_labels_available = "label_stroke" in working.columns
        self._accumulate_risk_factors(working, grouped, group_field)
        for name, values in (model_scores or {}).items():
            working[f"_{name}"] = values

        for region, group in working.groupby(group_field, dropna=False):
            item = grouped[str(region)]
            item["residents"] = int(len(group))
            if "label_heart" in group:
                item["heart_positive"] = int(pd.to_numeric(group["label_heart"], errors="coerce").fillna(0).sum())
            if "label_stroke" in group:
                item["stroke_positive"] = int(pd.to_numeric(group["label_stroke"], errors="coerce").fillna(0).sum())
            if "_heart_probability" in group:
                item["heart_high"] = int((group["_heart_probability"] >= 0.60).sum())
            if "_stroke_probability" in group:
                item["stroke_high"] = int((group["_stroke_probability"] >= 0.60).sum())
            heart_labels = self._series_or_zero(group, "label_heart")
            stroke_labels = self._series_or_zero(group, "label_stroke")
            item["comorbidity"] = int(
                ((heart_labels == 1) & (stroke_labels == 1)).sum()
            )

        rows = []
        for region, values in sorted(grouped.items(), key=lambda item: item[1]["residents"], reverse=True):
            suppressed = values["residents"] < self.min_group_size
            residents = values["residents"] if not suppressed else None
            denominator = values["residents"] or 1
            rows.append(
                {
                    "region": region,
                    "district": region if coverage["coverage_level"] == "district" else None,
                    "community": region if coverage["coverage_level"] == "community" else None,
                    "residents": residents,
                    "heart_rate": None if suppressed or not heart_labels_available else round(values["heart_positive"] / denominator * 100, 2),
                    "stroke_rate": None if suppressed or not stroke_labels_available else round(values["stroke_positive"] / denominator * 100, 2),
                    "heart_high_risk_rate": None if suppressed else round(values["heart_high"] / denominator * 100, 2),
                    "stroke_high_risk_rate": None if suppressed else round(values["stroke_high"] / denominator * 100, 2),
                    "comorbidity_rate": None if suppressed or not (heart_labels_available and stroke_labels_available) else round(values["comorbidity"] / denominator * 100, 2),
                    "suppressed": suppressed,
                }
            )
        return {"coverage": coverage, "regions": rows}

    def _accumulate_risk_factors(self, dataframe, grouped, group_field):
        # 初始化结构，具体统计在 dashboard 级别重新计算，保持聚合字段简单。
        for region in dataframe[group_field].dropna().astype(str).unique():
            grouped[region]["residents"] = 0

    @staticmethod
    def _values(dataframe, field):
        if field not in dataframe:
            return []
        return [value for value in dataframe[field].dropna().astype(str).unique().tolist() if value.strip()]

    @staticmethod
    def _first(values):
        return values[0] if values else None

    @staticmethod
    def _is_chengdu(values):
        return any("成都" in value for value in values)

    @staticmethod
    def _series_or_zero(dataframe, field):
        if field not in dataframe:
            return pd.Series(0, index=dataframe.index)
        return pd.to_numeric(dataframe[field], errors="coerce").fillna(0)
