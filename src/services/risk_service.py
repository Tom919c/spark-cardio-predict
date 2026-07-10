from src.ml.predict import CardioRiskPredictor
from src.ml.train import CardioModelTrainer
from src.models.intervention_plan import InterventionPlan
from src.models.risk_result import RiskResult
from src.services.data_service import DataService
from src.services.intervention_service import InterventionService
from src.utils.risk_rules import build_indicator_insights, build_risk_interpretation
from src.utils.training_recorder import TrainingResultRecorder
from src.utils.validator import PredictionInputValidator


class RiskService:
    """Provides risk model training and prediction services."""

    def __init__(self, config):
        self.config = config
        self.data_service = DataService(config)
        self.training_recorder = TrainingResultRecorder()
        self.intervention_service = InterventionService()
        self.default_model_name = config.get("DEFAULT_MODEL_NAME", "logistic_regression")
        self.available_model_names = config.get("AVAILABLE_MODEL_NAMES", [])
        self.enable_multi_run_training = config.get("ENABLE_MULTI_RUN_TRAINING", True)
        self.default_training_rounds = config.get("DEFAULT_TRAINING_ROUNDS", 3)
        self.best_model_metric = config.get("BEST_MODEL_METRIC", "roc_auc")
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])

    def get_platform_summary(self):
        return {
            "target_groups": ["医疗机构", "健康管理机构", "中老年群体"],
            "core_capabilities": [
                "数据采集与整合",
                "风险预测建模",
                "风险等级评估",
                "干预方案生成",
                "趋势分析与可视化",
            ],
            "status": "training_ready",
            "default_model_name": self.default_model_name,
            "available_model_names": self.available_model_names,
            "multi_run_training": self.enable_multi_run_training,
            "default_training_rounds": self.default_training_rounds,
            "best_model_metric": self.best_model_metric,
        }

    def train_model(self, model_name=None, run_label="", rounds=None):
        selected_model_name = model_name or self.default_model_name
        selected_rounds = rounds or self.default_training_rounds
        if selected_rounds <= 0:
            raise ValueError("rounds 必须大于 0。")

        training_dataframe = self.data_service.get_training_dataframe()
        trainer = CardioModelTrainer(self.config)
        training_result = trainer.train_multiple_rounds(
            dataframe=training_dataframe,
            model_name=selected_model_name,
            rounds=selected_rounds,
            run_label=run_label,
            best_metric=self.best_model_metric,
        )
        self.training_recorder.append_multi_round_result(training_result)
        return training_result

    def predict_risk(self, sample, model_path):
        validator = PredictionInputValidator(required_fields=self.feature_columns)
        validation_result = validator.validate_payload(sample)
        if not validation_result["valid"]:
            raise ValueError(
                f"Prediction payload is missing fields: {validation_result['missing_fields']}"
            )

        predictor = CardioRiskPredictor()
        prediction = predictor.predict(model_path, sample)
        risk_level = self._map_risk_level(prediction.get("predicted_probability"))
        interpretation = build_risk_interpretation(
            sample=sample,
            predicted_probability=prediction.get("predicted_probability"),
            risk_level=risk_level,
        )
        intervention_plan = InterventionPlan(
            risk_level=risk_level,
            suggestions=self.intervention_service.build_plan(sample, risk_level)[
                "suggestions"
            ],
        )
        risk_result = RiskResult(
            predicted_label=prediction.get("predicted_label"),
            predicted_probability=prediction.get("predicted_probability"),
            risk_level=risk_level,
            model_path=model_path,
            risk_summary=interpretation.get("risk_summary", ""),
            indicator_insights=build_indicator_insights(sample),
            intervention_plan=intervention_plan.to_dict(),
            key_highlights=interpretation.get("key_highlights", []),
        )
        result = risk_result.to_dict()
        result["required_fields"] = self.feature_columns
        result["risk_probability_percent"] = interpretation.get(
            "risk_probability_percent"
        )
        return result

    def _map_risk_level(self, probability):
        if probability is None:
            return "unknown"
        if probability < 0.35:
            return "low"
        if probability < 0.7:
            return "medium"
        return "high"
