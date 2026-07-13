"""群体健康分析服务：基于成都仿真人口数据提供大屏聚合统计。"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd


class PopulationService:
    """提供本地 CSV 聚合统计，接口与 Spark 群体分析任务对齐。"""

    _dataframe_cache = {}
    _cache_lock = Lock()

    REQUIRED_COLUMNS = [
        "age", "district", "age_group", "bmi",
        "hypertension", "diabetes", "cholesterol", "smoker", "exercise",
        "label_heart", "label_stroke",
    ]

    def __init__(self, config):
        self.population_path = Path(config["POPULATION_DATASET_PATH"])

    def dashboard(self):
        dataframe = self._dataframe
        total = len(dataframe)
        high_risk = dataframe[
            (dataframe["age"] >= 65)
            & ((dataframe["label_heart"] == 1) | (dataframe["label_stroke"] == 1))
        ]
        return {
            "data_source": str(self.population_path),
            "total_residents": int(total),
            "heart_risk_rate": self._percentage(dataframe["label_heart"]),
            "stroke_risk_rate": self._percentage(dataframe["label_stroke"]),
            "comorbidity_rate": self._percentage(
                (dataframe["label_heart"] == 1) & (dataframe["label_stroke"] == 1)
            ),
            "high_risk_follow_up_count": int(len(high_risk)),
            "districts": self._districts(dataframe),
            "age_groups": self._age_groups(dataframe),
            "risk_factors": self._risk_factors(dataframe),
        }

    def follow_up_list(self, limit=100):
        dataframe = self._dataframe
        screened = dataframe[
            (dataframe["age"] >= 65)
            & ((dataframe["label_heart"] == 1) | (dataframe["label_stroke"] == 1))
        ].copy()
        screened["risk_score"] = (
            screened["label_heart"]
            + screened["label_stroke"]
            + screened["hypertension"]
            + screened["diabetes"]
            + (screened["cholesterol"] >= 2).astype(int)
        )
        result = screened.sort_values(["risk_score", "age"], ascending=False).head(limit)
        records = result.to_dict(orient="records")
        for record in records:
            record["screening_basis"] = "年龄>=65 且仿真双风险标签至少一项为 1"
        return records

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
                chunks = pd.read_csv(
                    self.population_path,
                    usecols=self.REQUIRED_COLUMNS,
                    chunksize=250_000,
                )
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
                "residents": int(row.residents),
                "heart_rate": round(float(row.heart_rate) * 100, 2),
                "stroke_rate": round(float(row.stroke_rate) * 100, 2),
                "comorbidity_rate": round(float(row.comorbidity_rate) * 100, 2),
            }
            for row in grouped.sort_values("stroke_rate", ascending=False).itertuples()
        ]

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
