from src.ml.predict import CardioRiskPredictor
from src.ml.train import CardioModelTrainer
from src.services.data_service import DataService
from src.utils.training_recorder import TrainingResultRecorder


class RiskService:
    """Provides risk model training and prediction services."""

    def __init__(self, config):
        self.config = config
        self.data_service = DataService(config)
        self.training_recorder = TrainingResultRecorder()
        self.default_model_name = config.get("DEFAULT_MODEL_NAME", "logistic_regression")
        self.available_model_names = config.get("AVAILABLE_MODEL_NAMES", [])
        self.enable_multi_run_training = config.get("ENABLE_MULTI_RUN_TRAINING", True)
        self.default_training_rounds = config.get("DEFAULT_TRAINING_ROUNDS", 3)
        self.best_model_metric = config.get("BEST_MODEL_METRIC", "roc_auc")

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
        predictor = CardioRiskPredictor()
        prediction = predictor.predict(model_path, sample)
        prediction["risk_level"] = self._map_risk_level(
            prediction.get("predicted_probability")
        )
        return prediction

    def _map_risk_level(self, probability):
        if probability is None:
            return "unknown"
        if probability < 0.35:
            return "low"
        if probability < 0.7:
            return "medium"
        return "high"
