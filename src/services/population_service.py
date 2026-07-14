"""群体健康分析服务：基于成都仿真人口数据提供大屏聚合统计。"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd

from src.services.region_service import RegionService
from src.utils.privacy import mask_records


class PopulationService:
    """提供本地 CSV 聚合统计，接口与 Spark 群体分析任务对齐。"""

    _dataframe_cache = {}
    _cache_lock = Lock()

    REQUIRED_COLUMNS = [
        "age", "district", "age_group", "bmi",
        "hypertension", "diabetes", "cholesterol", "smoker", "exercise",
        "label_heart", "label_stroke",
    ]
    OPTIONAL_REGION_COLUMNS = [
        "province", "city", "region", "community", "community_id", "risk_cluster"
    ]

    def __init__(self, config):
        self.population_path = Path(config["POPULATION_DATASET_PATH"])
        self.min_group_size = int(config.get("PRIVACY_MIN_GROUP_SIZE", 5))

    def dashboard(self):
        dataframe = self._dataframe
        coverage = RegionService(self.min_group_size).infer_coverage(dataframe)
        total = len(dataframe)
        high_risk_mask = self._screening_high_risk_mask(dataframe)
        follow_up_mask = self._follow_up_mask(dataframe)
        return {
            "data_source": self.population_path.name,
            "total_residents": int(total),
            "heart_risk_rate": self._percentage(dataframe["label_heart"]),
            "stroke_risk_rate": self._percentage(dataframe["label_stroke"]),
            "comorbidity_rate": self._percentage(
                (dataframe["label_heart"] == 1) & (dataframe["label_stroke"] == 1)
            ),
            "high_risk_count": int(high_risk_mask.sum()),
            "high_risk_follow_up_count": int(follow_up_mask.sum()),
            "districts": self._districts(dataframe),
            "age_groups": self._age_groups(dataframe),
            "risk_factors": self._risk_factors(dataframe),
            "coverage": coverage,
            "coverage_level": coverage["coverage_level"],
            "coverage_name": coverage["coverage_name"],
            "map_name": coverage["map_name"],
            "map_available": coverage["map_available"],
            "risk_metric_source": "labels",
        }

    def follow_up_list(self, limit=100):
        dataframe = self._dataframe
        screened = dataframe.loc[self._follow_up_mask(dataframe)].copy()
        privacy_field = self._privacy_group_field(dataframe)
        if privacy_field:
            group_sizes = dataframe[privacy_field].fillna("未知").astype(str).value_counts()
            screened = screened[
                screened[privacy_field]
                .fillna("未知")
                .astype(str)
                .map(group_sizes)
                .ge(self.min_group_size)
            ]
        screened["risk_score"] = (
            1.10 * screened["label_heart"]
            + 1.10 * screened["label_stroke"]
            + 0.55 * screened["hypertension"]
            + 0.55 * screened["diabetes"]
            + 0.35 * (screened["cholesterol"] >= 2).astype(int)
            + 0.35 * (screened["smoker"] == 2).astype(int)
            + 0.35 * (screened["bmi"] >= 30).astype(int)
            + 0.65 * screened.get("risk_cluster", 0)
        ).clip(upper=5.0).round(3)
        result = screened.sort_values(
            ["risk_score", "bmi", "age"], ascending=False
        ).head(limit)
        records = result.to_dict(orient="records")
        for record in records:
            record["screening_basis"] = "事件标签或多项危险因素聚集"
        return mask_records(records)

    @staticmethod
    def _follow_up_mask(dataframe):
        """筛出需要重点随访的人群，避免名单被年龄单一规则垄断。"""
        heart = pd.to_numeric(dataframe["label_heart"], errors="coerce").fillna(0)
        stroke = pd.to_numeric(dataframe["label_stroke"], errors="coerce").fillna(0)
        factors = PopulationService._risk_factor_count(dataframe)
        event_label = (heart == 1) | (stroke == 1)
        multi_factor = factors >= 4
        if "risk_cluster" in dataframe:
            cluster = pd.to_numeric(dataframe["risk_cluster"], errors="coerce").fillna(0)
            cluster_priority = (cluster == 1) & (factors >= 3)
        else:
            cluster_priority = pd.Series(False, index=dataframe.index)
        return event_label | multi_factor | cluster_priority

    @classmethod
    def _screening_high_risk_mask(cls, dataframe):
        """筛查高危口径：年度事件代理或至少四项危险因素聚集。"""
        event_label = (
            (pd.to_numeric(dataframe["label_heart"], errors="coerce").fillna(0) == 1)
            | (pd.to_numeric(dataframe["label_stroke"], errors="coerce").fillna(0) == 1)
        )
        return event_label | (cls._risk_factor_count(dataframe) >= 4)

    @staticmethod
    def _risk_factor_count(dataframe):
        return (
            pd.to_numeric(dataframe["hypertension"], errors="coerce").fillna(0)
            + pd.to_numeric(dataframe["diabetes"], errors="coerce").fillna(0)
            + (pd.to_numeric(dataframe["cholesterol"], errors="coerce").fillna(0) >= 2).astype(int)
            + (pd.to_numeric(dataframe["smoker"], errors="coerce").fillna(0) == 2).astype(int)
            + (pd.to_numeric(dataframe["bmi"], errors="coerce").fillna(0) >= 30).astype(int)
        )

    @property
    def _dataframe(self):
        if not self.population_path.exists():
            raise FileNotFoundError(
                "未找到成都市仿真数据，请先运行 scripts/generate_chengdu_health_data.py。"
            )
        stat = self.population_path.stat()
        cache_key = str(self.population_path.resolve())
        cache_version = (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            cached = self._dataframe_cache.get(cache_key)
            if cached and cached[0] == cache_version:
                return cached[1]

            # 分块读取并只保留大屏列，避免一次加载大量行占用过多内存。
            try:
                header = pd.read_csv(self.population_path, nrows=0)
                missing = [column for column in self.REQUIRED_COLUMNS if column not in header.columns]
                if missing:
                    raise ValueError(f"字段不完整: {missing}")
                usecols = self.REQUIRED_COLUMNS + [
                    column for column in self.OPTIONAL_REGION_COLUMNS
                    if column in header.columns
                ]
                chunks = pd.read_csv(self.population_path, usecols=usecols, chunksize=250_000)
                dataframe = pd.concat(chunks, ignore_index=True)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "仿真数据读取失败或字段不完整，请重新生成成都居民数据。"
                ) from exc
            self._dataframe_cache[cache_key] = (cache_version, dataframe)
            return dataframe

    def _districts(self, dataframe):
        grouped = dataframe.assign(
            comorbidity=(dataframe["label_heart"] == 1)
            & (dataframe["label_stroke"] == 1)
        ).groupby("district", as_index=False).agg(
            residents=("age", "size"),
            heart_rate=("label_heart", "mean"),
            stroke_rate=("label_stroke", "mean"),
            comorbidity_rate=("comorbidity", "mean"),
        )
        return [
            {
                "district": row.district,
                "residents": None if int(row.residents) < self.min_group_size else int(row.residents),
                "heart_rate": None if int(row.residents) < self.min_group_size else round(float(row.heart_rate) * 100, 2),
                "stroke_rate": None if int(row.residents) < self.min_group_size else round(float(row.stroke_rate) * 100, 2),
                "comorbidity_rate": None if int(row.residents) < self.min_group_size else round(float(row.comorbidity_rate) * 100, 2),
                "suppressed": int(row.residents) < self.min_group_size,
            }
            for row in grouped.sort_values("stroke_rate", ascending=False).itertuples()
        ]

    @staticmethod
    def _privacy_group_field(dataframe):
        for field in ("community_id", "community", "district"):
            if field in dataframe.columns:
                return field
        return None

    def _age_groups(self, dataframe):
        grouped = dataframe.groupby("age_group", observed=True).agg(
            residents=("age", "size"),
            heart=("label_heart", "sum"),
            stroke=("label_stroke", "sum"),
        )
        return [
            {
                "age_group": str(index),
                "residents": int(row.residents),
                "heart": int(row.heart),
                "stroke": int(row.stroke),
            }
            for index, row in grouped.iterrows()
        ]

    def _risk_factors(self, dataframe):
        factors = {
            "高血压": dataframe["hypertension"].mean(),
            "糖尿病": dataframe["diabetes"].mean(),
            "血脂异常": (dataframe["cholesterol"] >= 2).mean(),
            "当前吸烟": (dataframe["smoker"] == 2).mean(),
            "超重或肥胖": (dataframe["bmi"] >= 25).mean(),
            "缺乏规律运动": (dataframe["exercise"] == 0).mean(),
        }
        return [
            {"name": name, "rate": round(float(rate) * 100, 2)}
            for name, rate in sorted(factors.items(), key=lambda item: item[1], reverse=True)
        ]

    @staticmethod
    def _percentage(values):
        return round(float(values.mean()) * 100, 2)
