from src.ml.model_explainer import ModelExplainer
from src.ml.model_registry import ModelRegistry
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
    """Provides merged dual-model training and prediction services."""

    def __init__(self, config):
        self.config = config
        self.data_service = DataService(config)
        self.training_recorder = TrainingResultRecorder()
        self.intervention_service = InterventionService()
        self.model_explainer = ModelExplainer()
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])
        self.thresholds = config.get("RISK_LEVEL_THRESHOLDS", (0.15, 0.35, 0.60, 0.80))
        self.default_training_rounds = config.get("DEFAULT_TRAINING_ROUNDS", 1)
        self.registry = ModelRegistry(
            config.get("MODEL_OUTPUT_DIR", "data/feature/models"),
            config.get("MODEL_MANIFEST_PATH", "data/feature/models/active_models.json"),
            self.feature_columns,
        )

    def get_platform_summary(self):
        return {
            "stage": "merged-platform",
            "target_groups": ["medical_institutions", "health_management", "residents"],
            "risk_models": ["heart", "stroke"],
            "required_features": self.feature_columns,
            "data_mode": self.config.get("DATA_MODE", "local"),
            "default_training_rounds": self.default_training_rounds,
            "model_status": self._model_status(),
        }

    def train_model(self, model_name=None, run_label="", rounds=None):
        del model_name
        selected_rounds = rounds or self.default_training_rounds
        if selected_rounds <= 0:
            raise ValueError("rounds must be greater than 0.")

        training_dataframe = self.data_service.get_training_dataframe()
        result = None
        for round_index in range(1, selected_rounds + 1):
            label = run_label or "phase1"
            round_label = f"{label}_round{round_index}" if selected_rounds > 1 else label
            result = CardioModelTrainer(self.config).train(training_dataframe, round_label)

        model_paths = {
            target: target_result["model_path"]
            for target, target_result in result["targets"].items()
        }
        metrics = {
            target: target_result["metrics"]
            for target, target_result in result["targets"].items()
        }
        result["registry"] = self.registry.register(model_paths, metrics)
        self.training_recorder.append_multi_round_result(result)
        return result

    def predict_risk(self, sample, heart_model_path=None, stroke_model_path=None):
        validator = PredictionInputValidator(required_fields=self.feature_columns)
        validation_result = validator.validate_payload(sample)
        if not validation_result["valid"]:
            raise ValueError(
                f"Prediction payload is missing fields: {validation_result['missing_fields']}"
            )

        normalised_sample = self._normalise_sample(sample)
        if heart_model_path and stroke_model_path:
            model_paths = {"heart": heart_model_path, "stroke": stroke_model_path}
        else:
            model_paths = self.registry.load_active_models()

        predictor = CardioRiskPredictor()
        prediction = predictor.predict(model_paths, normalised_sample)
        heart_explanation = self.model_explainer.explain_prediction(
            model_path=model_paths["heart"],
            sample=normalised_sample,
        )
        stroke_explanation = self.model_explainer.explain_prediction(
            model_path=model_paths["stroke"],
            sample=normalised_sample,
        )
        final_category = self._build_final_category(
            prediction["heart_predicted_probability"],
            prediction["stroke_predicted_probability"],
        )
        risk_level = self._risk_level(
            max(
                prediction["heart_predicted_probability"],
                prediction["stroke_predicted_probability"],
            )
        )
        interpretation = build_risk_interpretation(
            sample=normalised_sample,
            heart_probability=prediction["heart_predicted_probability"],
            stroke_probability=prediction["stroke_predicted_probability"],
            final_category=final_category,
        )
        intervention_plan = InterventionPlan(
            risk_level=final_category,
            suggestions=self.intervention_service.build_plan(normalised_sample, final_category)[
                "suggestions"
            ],
        )

        risk_result = RiskResult(
            predicted_label=final_category,
            risk_level=final_category,
            model_path="dual_model",
            risk_summary=interpretation["risk_summary"],
            indicator_insights=build_indicator_insights(normalised_sample),
            intervention_plan=intervention_plan.to_dict(),
            key_highlights=interpretation["key_highlights"],
        )
        result = risk_result.to_dict()
        result["required_fields"] = self.feature_columns
        result["heart_model_path"] = model_paths["heart"]
        result["stroke_model_path"] = model_paths["stroke"]
        result["heart_predicted_label"] = prediction["heart_predicted_label"]
        result["stroke_predicted_label"] = prediction["stroke_predicted_label"]
        result["heart_predicted_probability"] = prediction["heart_predicted_probability"]
        result["stroke_predicted_probability"] = prediction["stroke_predicted_probability"]
        result["heart_probability_percent"] = interpretation["heart_probability_percent"]
        result["stroke_probability_percent"] = interpretation["stroke_probability_percent"]
        result["final_category"] = final_category
        result["risk_level"] = risk_level
        result["heart_shap_explanation"] = heart_explanation
        result["stroke_shap_explanation"] = stroke_explanation
        result["combined_shap_summary"] = self._build_combined_shap_summary(
            heart_explanation=heart_explanation,
            stroke_explanation=stroke_explanation,
        )
        return result

    def _build_final_category(self, heart_probability, stroke_probability):
        heart_positive = heart_probability is not None and heart_probability >= self.thresholds[1]
        stroke_positive = (
            stroke_probability is not None and stroke_probability >= self.thresholds[1]
        )
        if not heart_positive and not stroke_positive:
            return 0
        if heart_positive and not stroke_positive:
            return 1
        if not heart_positive and stroke_positive:
            return 2
        return 3

    def _risk_level(self, probability):
        if probability < self.thresholds[0]:
            return {"code": 1, "name": "I", "color": "#16a34a"}
        if probability < self.thresholds[1]:
            return {"code": 2, "name": "II", "color": "#ca8a04"}
        if probability < self.thresholds[2]:
            return {"code": 3, "name": "III", "color": "#ea580c"}
        if probability < self.thresholds[3]:
            return {"code": 4, "name": "IV", "color": "#dc2626"}
        return {"code": 5, "name": "V", "color": "#b91c1c"}

    def _build_combined_shap_summary(self, heart_explanation, stroke_explanation):
        if not heart_explanation.get("available") or not stroke_explanation.get("available"):
            return {
                "available": False,
                "message": "SHAP explanation is unavailable for one or more models.",
                "top_positive_factors": [],
                "top_negative_factors": [],
            }

        merged = {}
        for source_name, explanation in (
            ("heart", heart_explanation),
            ("stroke", stroke_explanation),
        ):
            for item in explanation.get("top_positive_factors", []) + explanation.get(
                "top_negative_factors", []
            ):
                feature_name = item["feature"]
                if feature_name not in merged:
                    merged[feature_name] = {
                        "feature": feature_name,
                        "feature_value": item.get("feature_value"),
                        "contribution": 0.0,
                        "impact_percent": 0.0,
                        "sources": [],
                    }
                merged[feature_name]["contribution"] += item.get("contribution", 0.0)
                merged[feature_name]["impact_percent"] += item.get("impact_percent", 0.0)
                merged[feature_name]["sources"].append(source_name)

        items = list(merged.values())
        positive = sorted(
            [item for item in items if item["contribution"] > 0],
            key=lambda item: item["contribution"],
            reverse=True,
        )[:4]
        negative = sorted(
            [item for item in items if item["contribution"] < 0],
            key=lambda item: item["contribution"],
        )[:4]

        for item in positive + negative:
            item["contribution"] = round(item["contribution"], 6)
            item["impact_percent"] = round(item["impact_percent"], 2)
            item["direction"] = "increase" if item["contribution"] >= 0 else "decrease"

        return {
            "available": True,
            "message": "Combined SHAP summary generated successfully.",
            "top_positive_factors": positive,
            "top_negative_factors": negative,
        }

    def _model_status(self):
        try:
            models = self.registry.load_active_models()
        except FileNotFoundError:
            return {"ready": False, "models": {}}
        return {"ready": True, "models": models}

    @staticmethod
    def _normalise_sample(sample):
        normalised = dict(sample)
        for field in (
            "age",
            "gender",
            "cholesterol",
            "diabetes",
            "hypertension",
            "smoker",
            "alcohol",
            "exercise",
        ):
            normalised[field] = int(float(normalised[field]))
        normalised["bmi"] = round(float(normalised["bmi"]), 2)
        return normalised
