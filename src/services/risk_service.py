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
    """Provides dual-model training and prediction services."""

    def __init__(self, config):
        self.config = config
        self.data_service = DataService(config)
        self.training_recorder = TrainingResultRecorder()
        self.intervention_service = InterventionService()
        self.default_model_name = config.get("DEFAULT_MODEL_NAME", "random_forest")
        self.available_model_names = config.get("AVAILABLE_MODEL_NAMES", [])
        self.enable_multi_run_training = config.get("ENABLE_MULTI_RUN_TRAINING", True)
        self.default_training_rounds = config.get("DEFAULT_TRAINING_ROUNDS", 3)
        self.best_model_metric = config.get("BEST_MODEL_METRIC", "roc_auc")
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])
        self.heart_risk_threshold = config.get("HEART_RISK_THRESHOLD", 0.5)
        self.stroke_risk_threshold = config.get("STROKE_RISK_THRESHOLD", 0.5)

    def get_platform_summary(self):
        return {
            "target_groups": ["医疗机构", "健康管理机构", "中老年群体"],
            "core_capabilities": [
                "数据采集与整合",
                "双模型风险预测建模",
                "联合风险等级评估",
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

    def predict_risk(self, sample, heart_model_path, stroke_model_path):
        validator = PredictionInputValidator(required_fields=self.feature_columns)
        validation_result = validator.validate_payload(sample)
        if not validation_result["valid"]:
            raise ValueError(
                f"Prediction payload is missing fields: {validation_result['missing_fields']}"
            )

        predictor = CardioRiskPredictor()
        prediction = predictor.predict(heart_model_path, stroke_model_path, sample)
        final_category = self._build_final_category(
            prediction["heart_predicted_probability"],
            prediction["stroke_predicted_probability"],
        )
        interpretation = build_risk_interpretation(
            sample=sample,
            heart_probability=prediction["heart_predicted_probability"],
            stroke_probability=prediction["stroke_predicted_probability"],
            final_category=final_category,
        )
        intervention_plan = InterventionPlan(
            risk_level=final_category,
            suggestions=self.intervention_service.build_plan(sample, final_category)[
                "suggestions"
            ],
        )

        risk_result = RiskResult(
            predicted_label=final_category,
            risk_level=final_category,
            model_path="dual_model",
            risk_summary=interpretation["risk_summary"],
            indicator_insights=build_indicator_insights(sample),
            intervention_plan=intervention_plan.to_dict(),
            key_highlights=interpretation["key_highlights"],
        )
        result = risk_result.to_dict()
        result["required_fields"] = self.feature_columns
        result["heart_model_path"] = heart_model_path
        result["stroke_model_path"] = stroke_model_path
        result["heart_predicted_label"] = prediction["heart_predicted_label"]
        result["stroke_predicted_label"] = prediction["stroke_predicted_label"]
        result["heart_predicted_probability"] = prediction["heart_predicted_probability"]
        result["stroke_predicted_probability"] = prediction["stroke_predicted_probability"]
        result["heart_probability_percent"] = interpretation["heart_probability_percent"]
        result["stroke_probability_percent"] = interpretation["stroke_probability_percent"]
        result["final_category"] = final_category
        return result

    def _build_final_category(self, heart_probability, stroke_probability):
        heart_positive = heart_probability is not None and heart_probability >= self.heart_risk_threshold
        stroke_positive = stroke_probability is not None and stroke_probability >= self.stroke_risk_threshold
        if not heart_positive and not stroke_positive:
            return 0
        if heart_positive and not stroke_positive:
            return 1
        if not heart_positive and stroke_positive:
            return 2
        return 3
