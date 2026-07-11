"""Application service for the two independent cardiovascular risk models."""

from __future__ import annotations

from src.ml.model_registry import ModelRegistry
from src.ml.predict import CardioRiskPredictor
from src.ml.train import CardioModelTrainer
from src.services.data_service import DataService
from src.services.intervention_service import InterventionService
from src.utils.risk_rules import build_indicator_insights
from src.utils.validator import PredictionInputValidator


class RiskService:
    """Coordinates training, model selection, prediction, and risk interpretation."""

    def __init__(self, config):
        self.config = config
        self.data_service = DataService(config)
        self.feature_columns = config["CARDIO_FEATURE_COLUMNS"]
        self.thresholds = config["RISK_LEVEL_THRESHOLDS"]
        self.registry = ModelRegistry(
            config["MODEL_OUTPUT_DIR"],
            config["MODEL_MANIFEST_PATH"],
            self.feature_columns,
        )
        self.intervention_service = InterventionService()

    def get_platform_summary(self):
        return {
            "stage": "phase_1",
            "data_mode": self.config["DATA_MODE"],
            "risk_models": ["heart", "stroke"],
            "required_features": self.feature_columns,
            "model_status": self._model_status(),
        }

    def train_models(self, run_label="phase1"):
        dataframe = self.data_service.get_training_dataframe()
        training_result = CardioModelTrainer(self.config).train(dataframe, run_label)
        model_paths = {
            target: result["model_path"]
            for target, result in training_result["targets"].items()
        }
        metrics = {
            target: result["metrics"]
            for target, result in training_result["targets"].items()
        }
        registry = self.registry.register(model_paths, metrics)
        return {**training_result, "registry": registry}

    def predict_risk(self, sample):
        validator = PredictionInputValidator(required_fields=self.feature_columns)
        validation = validator.validate_payload(sample)
        if not validation["valid"]:
            raise ValueError(f"缺少必填字段: {validation['missing_fields']}")

        normalised_sample = self._normalise_sample(sample)
        model_paths = self.registry.load_active_models()
        # 模型路径只由后端清单管理，避免浏览器传入任意本地文件路径。
        prediction = CardioRiskPredictor().predict(model_paths, normalised_sample)
        heart_probability = prediction["heart_predicted_probability"]
        stroke_probability = prediction["stroke_predicted_probability"]
        max_probability = max(heart_probability, stroke_probability)
        risk_level = self._risk_level(max_probability)

        return {
            "heart_probability": round(heart_probability, 4),
            "stroke_probability": round(stroke_probability, 4),
            "heart_probability_percent": round(heart_probability * 100, 2),
            "stroke_probability_percent": round(stroke_probability * 100, 2),
            "heart_predicted_label": prediction["heart_predicted_label"],
            "stroke_predicted_label": prediction["stroke_predicted_label"],
            "max_probability": round(max_probability, 4),
            "risk_level": risk_level,
            "risk_summary": self._risk_summary(risk_level, heart_probability, stroke_probability),
            "indicator_insights": build_indicator_insights(normalised_sample),
            "intervention_plan": self.intervention_service.build_plan(
                normalised_sample, risk_level
            ),
        }

    def _model_status(self):
        try:
            models = self.registry.load_active_models()
        except FileNotFoundError:
            return {"ready": False, "models": {}}
        return {"ready": True, "models": models}

    def _risk_level(self, probability):
        # 综合等级取两项独立事件概率的较高值，对应项目方案的医学 V 级规则。
        if probability < self.thresholds[0]:
            return {"code": 1, "name": "I级：健康", "color": "#16a34a"}
        if probability < self.thresholds[1]:
            return {"code": 2, "name": "II级：关注", "color": "#ca8a04"}
        if probability < self.thresholds[2]:
            return {"code": 3, "name": "III级：中危", "color": "#ea580c"}
        if probability < self.thresholds[3]:
            return {"code": 4, "name": "IV级：高危", "color": "#dc2626"}
        return {"code": 5, "name": "V级：极高危", "color": "#b91c1c"}

    @staticmethod
    def _risk_summary(risk_level, heart_probability, stroke_probability):
        dominant = "心脏事件" if heart_probability >= stroke_probability else "脑卒中"
        return f"当前以{dominant}风险为主，综合评估为{risk_level['name']}。"

    @staticmethod
    def _normalise_sample(sample):
        normalised = dict(sample)
        for field in ("age", "gender", "cholesterol", "diabetes", "hypertension", "smoker", "alcohol", "exercise"):
            normalised[field] = int(float(normalised[field]))
        normalised["bmi"] = round(float(normalised["bmi"]), 2)
        return normalised
