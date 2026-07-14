"""阶段二本地分析引擎：分块读取 CSV，输出与 Spark ADS 对齐的结果。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.services.batch_risk_service import BatchRiskService
from src.services.region_service import RegionService
from src.utils.field_mapping import canonicalize_pandas_columns
from src.utils.privacy import mask_records


class Phase2AnalysisService:
    """本地模式的可运行分析引擎，HDFS 模式由 Spark 作业复用同一结果契约。"""

    def __init__(self, config):
        self.config = config
        self.chunk_size = int(config.get("LOCAL_ANALYSIS_CHUNK_SIZE", 100_000))
        self.region_service = RegionService(config.get("PRIVACY_MIN_GROUP_SIZE", 5))
        self.batch_service = BatchRiskService(config)

    def analyze_file(self, input_path, result_dir, metadata=None):
        input_path = Path(input_path)
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        metadata = metadata or {}
        if not input_path.exists():
            raise FileNotFoundError("分析数据文件不存在。")

        first_chunk = None
        geo_samples = []
        total_rows = 0
        invalid_rows = 0
        labels_available = True
        group_stats = defaultdict(self._new_group)
        age_groups = defaultdict(self._new_age_group)
        factor_counts = defaultdict(int)
        prediction_enabled = False
        high_risk_follow_up_count = 0
        follow_up_candidates = []
        model_paths = None
        field_mapping = {}
        try:
            model_paths = self.batch_service.registry.load_active_models()
        except FileNotFoundError:
            model_paths = None

        for chunk in pd.read_csv(
            input_path,
            chunksize=self.chunk_size,
            encoding=self.config.get("DATASET_ENCODING", "utf-8"),
            sep=self.config.get("DATASET_SEPARATOR", ","),
        ):
            chunk, current_mapping = canonicalize_pandas_columns(chunk)
            if not field_mapping:
                field_mapping = current_mapping
            if first_chunk is None:
                first_chunk = chunk.copy()
                self._validate_columns(chunk.columns)
                labels_available = {"label_heart", "label_stroke"}.issubset(chunk.columns)
            total_rows += len(chunk)
            invalid_rows += self._count_invalid_rows(chunk)
            geo_samples.append(self._geo_sample(chunk))
            group_field = self._group_field(chunk)
            if group_field is None:
                chunk["_phase2_region"] = "全部数据"
                group_field = "_phase2_region"
            chunk[group_field] = chunk[group_field].fillna("未知").astype(str)
            for region, group in chunk.groupby(group_field, dropna=False):
                stats = group_stats[str(region)]
                stats["residents"] += len(group)
                if "label_heart" in group:
                    stats["heart_positive"] += self._sum_binary(group["label_heart"])
                if "label_stroke" in group:
                    stats["stroke_positive"] += self._sum_binary(group["label_stroke"])
                if "label_heart" in group and "label_stroke" in group:
                    stats["comorbidity"] += int(
                        ((self._numeric(group["label_heart"]) == 1)
                         & (self._numeric(group["label_stroke"]) == 1)).sum()
                    )

            self._accumulate_factors(chunk, factor_counts)
            self._accumulate_age_groups(chunk, age_groups)

            scored_this_chunk = False
            if model_paths and set(self.batch_service.feature_columns).issubset(chunk.columns):
                scored_this_chunk = True
                prediction_enabled = True
                prepared = self.batch_service._prepare_features(chunk)
                predictions = self.batch_service.predictor.predict_dataframe(model_paths, prepared)
                heart = pd.Series(predictions["heart_predicted_probability"], index=chunk.index)
                stroke = pd.Series(predictions["stroke_predicted_probability"], index=chunk.index)
                for region, group in chunk.groupby(group_field, dropna=False):
                    key = str(region)
                    group_stats[key]["heart_high"] += int((heart.loc[group.index] >= 0.60).sum())
                    group_stats[key]["stroke_high"] += int((stroke.loc[group.index] >= 0.60).sum())
                    group_stats[key]["predicted_comorbidity"] += int(
                        ((heart.loc[group.index] >= 0.60) & (stroke.loc[group.index] >= 0.60)).sum()
                    )
                high_risk_follow_up_count += int(((heart >= 0.60) | (stroke >= 0.60)).sum())
                candidate_mask = (heart >= 0.60) | (stroke >= 0.60)
                for index in chunk.index[candidate_mask.to_numpy()]:
                    score = max(float(heart.loc[index]), float(stroke.loc[index]))
                    source = chunk.loc[index]
                    follow_up_candidates.append(
                        {
                            "resident_id": self._json_scalar(source.get("resident_id", index)),
                            "name": self._json_scalar(source.get("name", "居民")),
                            "phone": self._json_scalar(source.get("phone")),
                            "age": self._json_scalar(source.get("age")),
                            "district": self._json_scalar(source.get("district", source.get("region", "未知"))),
                            "community": self._json_scalar(source.get("community", source.get("community_id"))),
                            "risk_score": round(score * 5, 3),
                            "heart_probability": round(float(heart.loc[index]), 6),
                            "stroke_probability": round(float(stroke.loc[index]), 6),
                            "screening_basis": "双模型概率达到高风险阈值",
                        }
                    )
                if "age_group" in chunk:
                    age_labels = chunk["age_group"].fillna("未知").astype(str)
                else:
                    age_labels = pd.cut(
                        self._numeric(chunk["age"]),
                        bins=[0, 44, 54, 64, 74, 200],
                        labels=["18-44", "45-54", "55-64", "65-74", "75+"],
                    ).astype(str)
                for age_label, age_frame in chunk.groupby(age_labels, dropna=False):
                    item = age_groups[str(age_label)]
                    item["heart"] += int((heart.loc[age_frame.index] >= 0.60).sum())
                    item["stroke"] += int((stroke.loc[age_frame.index] >= 0.60).sum())

            if not scored_this_chunk and labels_available and {"label_heart", "label_stroke"}.issubset(chunk.columns):
                heart_labels = self._numeric(chunk["label_heart"]) == 1
                stroke_labels = self._numeric(chunk["label_stroke"]) == 1
                high_risk_follow_up_count += int((heart_labels | stroke_labels).sum())

        if first_chunk is None:
            raise ValueError("CSV 文件为空。")
        coverage_frame = pd.concat(geo_samples, ignore_index=True) if geo_samples else first_chunk
        coverage = self.region_service.infer_coverage(coverage_frame)
        metric_available = prediction_enabled or labels_available
        use_labels = labels_available and not prediction_enabled
        districts = self._build_group_rows(
            group_stats, coverage, metric_available, use_labels
        )
        result = self._build_result(
            total_rows=total_rows,
            invalid_rows=invalid_rows,
            labels_available=labels_available,
            prediction_enabled=prediction_enabled,
            group_rows=districts,
            group_stats=group_stats,
            age_groups=age_groups,
            factor_counts=factor_counts,
            coverage=coverage,
            metadata=metadata,
            high_risk_follow_up_count=high_risk_follow_up_count,
            metric_available=metric_available,
            field_mapping=field_mapping,
            follow_up_candidates=follow_up_candidates,
            use_labels=use_labels,
        )
        (result_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(districts).to_csv(result_dir / "region_summary.csv", index=False, encoding="utf-8-sig")
        return result

    def _validate_columns(self, columns):
        if "age" not in columns:
            raise ValueError("上传数据至少需要 age 字段。")
        if not any(field in columns for field in ("province", "city", "region", "district", "community", "community_id")):
            raise ValueError("上传数据至少需要一个区域字段。")

    @staticmethod
    def _geo_sample(chunk):
        fields = [field for field in ("province", "city", "region", "district", "community", "community_id") if field in chunk]
        if not fields:
            return pd.DataFrame({})
        return chunk[fields].head(1000).drop_duplicates()

    @staticmethod
    def _group_field(chunk):
        if "district" in chunk:
            return "district"
        if "community_id" in chunk:
            return "community_id"
        if "community" in chunk:
            return "community"
        if "city" in chunk:
            return "city"
        if "region" in chunk:
            return "region"
        if "province" in chunk:
            return "province"
        return None

    @staticmethod
    def _new_group():
        return {
            "residents": 0,
            "heart_positive": 0,
            "stroke_positive": 0,
            "comorbidity": 0,
            "heart_high": 0,
            "stroke_high": 0,
            "predicted_comorbidity": 0,
        }

    @staticmethod
    def _new_age_group():
        return {"residents": 0, "heart": 0, "stroke": 0}

    def _build_group_rows(self, group_stats, coverage, metric_available, labels_available):
        rows = []
        for name, values in sorted(group_stats.items(), key=lambda item: item[1]["residents"], reverse=True):
            suppressed = values["residents"] < self.region_service.min_group_size
            denominator = values["residents"] or 1
            if labels_available:
                heart_numerator = values["heart_positive"]
                stroke_numerator = values["stroke_positive"]
                comorbidity = values["comorbidity"]
            else:
                heart_numerator = values["heart_high"]
                stroke_numerator = values["stroke_high"]
                comorbidity = values["predicted_comorbidity"]
            row = {
                "region": name,
                "district": name if coverage["coverage_level"] == "district" else None,
                "community": name if coverage["coverage_level"] == "community" else None,
                "residents": None if suppressed else values["residents"],
                "heart_rate": None if suppressed or not metric_available else round(heart_numerator / denominator * 100, 2),
                "stroke_rate": None if suppressed or not metric_available else round(stroke_numerator / denominator * 100, 2),
                "comorbidity_rate": None if suppressed or not metric_available else round(comorbidity / denominator * 100, 2),
                "suppressed": suppressed,
            }
            rows.append(row)
        return rows

    def _build_result(self, total_rows, invalid_rows, labels_available, prediction_enabled, group_rows, group_stats, age_groups, factor_counts, coverage, metadata, high_risk_follow_up_count, metric_available, field_mapping=None, follow_up_candidates=None, use_labels=False):
        heart_numerator = sum(
            item["heart_positive"] if use_labels else item["heart_high"]
            for item in group_stats.values()
        )
        stroke_numerator = sum(
            item["stroke_positive"] if use_labels else item["stroke_high"]
            for item in group_stats.values()
        )
        comorbidity_numerator = sum(
            item["comorbidity"] if use_labels else item["predicted_comorbidity"]
            for item in group_stats.values()
        )
        heart_rate = self._rate(heart_numerator, total_rows) if metric_available else None
        stroke_rate = self._rate(stroke_numerator, total_rows) if metric_available else None
        candidates = sorted(
            follow_up_candidates or [],
            key=lambda item: (float(item.get("risk_score", 0)), float(item.get("age", 0) or 0)),
            reverse=True,
        )[:500]
        return {
            "dataset_id": metadata.get("dataset_id"),
            "data_period": metadata.get("data_period"),
            "field_mapping": field_mapping or {},
            "total_residents": int(total_rows),
            "invalid_rows": int(invalid_rows),
            "labels_available": bool(use_labels),
            "risk_metric_source": "model_probability" if prediction_enabled else ("labels" if use_labels else "unavailable"),
            "model_version": self.batch_service._model_version() if prediction_enabled else "labels-only",
            "coverage_level": coverage["coverage_level"],
            "coverage_name": coverage["coverage_name"],
            "map_name": coverage["map_name"],
            "map_available": coverage["map_available"],
            "coverage": coverage,
            "districts": group_rows,
            "region_analysis": group_rows,
            "age_groups": [
                {"age_group": name, **values}
                for name, values in sorted(age_groups.items())
            ],
            "risk_factors": [
                {"name": name, "rate": round(value / total_rows * 100, 2) if total_rows else 0}
                for name, value in sorted(factor_counts.items(), key=lambda item: item[1], reverse=True)
            ],
            "heart_risk_rate": heart_rate,
            "stroke_risk_rate": stroke_rate,
            "comorbidity_rate": self._rate(comorbidity_numerator, total_rows) if metric_available else None,
            "high_risk_follow_up_count": int(high_risk_follow_up_count),
            "follow_ups": mask_records(candidates),
        }

    @staticmethod
    def _count_invalid_rows(chunk):
        """统计明显不合法的年龄记录，但保留原始行供后续审计。"""
        if "age" not in chunk:
            return len(chunk)
        ages = pd.to_numeric(chunk["age"], errors="coerce")
        return int(ages.isna().sum() + ((ages <= 0) | (ages > 120)).sum())

    @staticmethod
    def _json_scalar(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (int, float, str, bool)):
            return value
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    @staticmethod
    def _global_rate(rows, field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return round(sum(values) / len(values), 2) if values else None

    @staticmethod
    def _rate(numerator, denominator):
        return round(float(numerator) / denominator * 100, 2) if denominator else 0

    @staticmethod
    def _numeric(values):
        return pd.to_numeric(values, errors="coerce").fillna(0)

    def _sum_binary(self, values):
        return int(self._numeric(values).sum())

    def _accumulate_factors(self, chunk, counts):
        if "hypertension" in chunk:
            counts["高血压"] += self._sum_binary(chunk["hypertension"])
        if "diabetes" in chunk:
            counts["糖尿病"] += self._sum_binary(chunk["diabetes"])
        if "cholesterol" in chunk:
            counts["血脂异常"] += int((self._numeric(chunk["cholesterol"]) >= 2).sum())
        if "smoker" in chunk:
            counts["当前吸烟"] += int((self._numeric(chunk["smoker"]) == 2).sum())
        if "bmi" in chunk:
            counts["超重或肥胖"] += int((self._numeric(chunk["bmi"]) >= 25).sum())
        if "exercise" in chunk:
            counts["缺乏规律运动"] += int((self._numeric(chunk["exercise"]) == 0).sum())

    def _accumulate_age_groups(self, chunk, groups):
        if "age_group" in chunk:
            labels = chunk["age_group"].fillna("未知").astype(str)
        elif "age" in chunk:
            labels = pd.cut(
                self._numeric(chunk["age"]),
                bins=[0, 44, 54, 64, 74, 200],
                labels=["18-44", "45-54", "55-64", "65-74", "75+"],
            ).astype(str)
        else:
            return
        for label, group in chunk.groupby(labels, dropna=False):
            item = groups[str(label)]
            item["residents"] += len(group)
            if "label_heart" in group:
                item["heart"] += self._sum_binary(group["label_heart"])
            if "label_stroke" in group:
                item["stroke"] += self._sum_binary(group["label_stroke"])
