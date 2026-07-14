"""批量双模型评分服务，供上传分析和独立批量评估复用。"""

from __future__ import annotations

import hashlib

import pandas as pd

from src.ml.model_registry import ModelRegistry
from src.ml.predict import CardioRiskPredictor


class BatchRiskService:
    """对一批已通过字段校验的数据执行心脏/卒中并联评分。"""

    def __init__(self, config):
        self.config = config
        self.feature_columns = list(config.get("CARDIO_FEATURE_COLUMNS", []))
        self.thresholds = tuple(config.get("RISK_LEVEL_THRESHOLDS", (0.15, 0.35, 0.60, 0.80)))
        self.registry = ModelRegistry(
            config.get("MODEL_OUTPUT_DIR", "data/models"),
            config.get("MODEL_MANIFEST_PATH", "data/models/active_models.json"),
            self.feature_columns,
        )
        self.predictor = CardioRiskPredictor()

    def score(self, dataframe, include_records=False, max_records=500):
        missing = [column for column in self.feature_columns if column not in dataframe]
        if missing:
            raise ValueError(f"批量评分缺少模型特征列: {missing}")
        frame = self._prepare_features(dataframe)
        model_paths = self.registry.load_active_models()
        predictions = self.predictor.predict_dataframe(model_paths, frame)
        heart = pd.Series(predictions["heart_predicted_probability"], index=frame.index)
        stroke = pd.Series(predictions["stroke_predicted_probability"], index=frame.index)
        overall = pd.concat([heart, stroke], axis=1).max(axis=1)
        levels = overall.map(self._risk_level)
        result = {
            "rows": int(len(frame)),
            "heart_high_risk_rate": round(float((heart >= self.thresholds[2]).mean()) * 100, 2),
            "stroke_high_risk_rate": round(float((stroke >= self.thresholds[2]).mean()) * 100, 2),
            "comorbidity_rate": round(float(((heart >= self.thresholds[2]) & (stroke >= self.thresholds[2])).mean()) * 100, 2),
            "model_version": self._model_version(),
            "records": [],
        }
        if include_records:
            for index in frame.index[:max_records]:
                record_id = self._masked_record_id(dataframe.loc[index], index)
                result["records"].append(
                    {
                        "record_id": record_id,
                        "heart_probability": round(float(heart.loc[index]), 6),
                        "stroke_probability": round(float(stroke.loc[index]), 6),
                        "risk_level": levels.loc[index],
                    }
                )
        return result

    def _prepare_features(self, dataframe):
        frame = dataframe.loc[:, self.feature_columns].copy()
        for column in self.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.fillna(frame.median(numeric_only=True))
        frame = frame.fillna(0)
        frame["age"] = frame["age"].clip(18, 95).round().astype(int)
        frame["bmi"] = frame["bmi"].clip(10.3, 79.8).round(2)
        frame["cholesterol"] = frame["cholesterol"].clip(1, 3).round().astype(int)
        frame["smoker"] = frame["smoker"].clip(0, 2).round().astype(int)
        for column in ("gender", "diabetes", "hypertension", "alcohol", "exercise"):
            frame[column] = frame[column].clip(0, 1).round().astype(int)
        return frame

    def _risk_level(self, probability):
        if probability < self.thresholds[0]:
            return {"code": 1, "name": "I级：健康"}
        if probability < self.thresholds[1]:
            return {"code": 2, "name": "II级：关注"}
        if probability < self.thresholds[2]:
            return {"code": 3, "name": "III级：中危"}
        if probability < self.thresholds[3]:
            return {"code": 4, "name": "IV级：高危"}
        return {"code": 5, "name": "V级：极高危"}

    @staticmethod
    def _masked_record_id(row, index):
        source = row.get("resident_id", row.get("id", index))
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        return f"R-{digest[:10]}"

    def _model_version(self):
        try:
            manifest = self.registry.manifest_path.read_text(encoding="utf-8")
            import json

            return json.loads(manifest).get("updated_at", "unknown")
        except (OSError, ValueError, TypeError):
            return "unknown"
