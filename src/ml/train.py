from datetime import datetime
from pathlib import Path
import re

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ml.evaluate import ModelEvaluator


class CardioModelTrainer:
    """Trains dual random forest models for heart-risk and stroke-risk prediction."""

    def __init__(self, config):
        self.config = config
        self.test_size = config.get("TRAIN_TEST_SPLIT_RATIO", 0.2)
        self.base_random_state = config.get("RANDOM_STATE", 42)
        self.model_output_dir = Path(config.get("MODEL_OUTPUT_DIR", "data/feature/models"))
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])

    def train_multiple_rounds(self, dataframe, model_name, rounds, run_label, best_metric):
        round_results = []
        for round_index in range(1, rounds + 1):
            round_results.append(
                self.train_single_round(
                    dataframe=dataframe,
                    model_name=model_name,
                    round_index=round_index,
                    run_label=run_label,
                )
            )

        best_result = self._select_best_result(round_results, best_metric)
        return {
            "model_name": model_name,
            "strategy": "dual_random_forest",
            "rounds": rounds,
            "run_label": run_label or "",
            "best_metric": best_metric,
            "best_result": best_result,
            "all_round_results": round_results,
        }

    def train_single_round(self, dataframe, model_name, round_index, run_label=""):
        round_random_state = self.base_random_state + round_index - 1
        features = dataframe[self.feature_columns].copy()

        heart_result = self._train_binary_target(
            features=features,
            target=dataframe["heart_risk"],
            target_name="heart_risk",
            model_name=f"{model_name}_heart",
            round_index=round_index,
            run_label=run_label,
            round_random_state=round_random_state,
        )
        stroke_result = self._train_binary_target(
            features=features,
            target=dataframe["stroke_risk"],
            target_name="stroke_risk",
            model_name=f"{model_name}_stroke",
            round_index=round_index,
            run_label=run_label,
            round_random_state=round_random_state,
        )

        combined_score = round(
            (
                heart_result["metrics"].get("roc_auc", 0)
                + stroke_result["metrics"].get("roc_auc", 0)
            )
            / 2,
            4,
        )

        return {
            "round_index": round_index,
            "round_random_state": round_random_state,
            "model_name": model_name,
            "run_label": run_label or "",
            "combined_score": combined_score,
            "heart_model_result": heart_result,
            "stroke_model_result": stroke_result,
        }

    def _train_binary_target(
        self, features, target, target_name, model_name, round_index, run_label, round_random_state
    ):
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=round_random_state,
            stratify=target,
        )

        model = self._build_model(round_random_state)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        probabilities = self._predict_probabilities(model, x_test)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_binary(y_test, predictions, probabilities)

        run_id = self._build_run_id(
            model_name=model_name,
            run_label=run_label,
            round_index=round_index,
        )
        model_path = self.save_model(model=model, run_id=run_id)

        return {
            "target_name": target_name,
            "run_id": run_id,
            "model_path": str(model_path),
            "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
            "metrics": metrics,
        }

    def save_model(self, model, run_id):
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_output_dir / f"{run_id}.joblib"
        joblib.dump(model, model_path)
        return model_path

    def _build_model(self, round_random_state):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=round_random_state,
        )

    def _predict_probabilities(self, model, features):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]
        return None

    def _build_run_id(self, model_name, run_label, round_index):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_label = self._normalize_run_label(run_label)
        if normalized_label:
            return f"{model_name}_{normalized_label}_round{round_index}_{timestamp}"
        return f"{model_name}_round{round_index}_{timestamp}"

    def _normalize_run_label(self, run_label):
        if not run_label:
            return ""
        normalized = re.sub(r"[^0-9A-Za-z_-]+", "_", run_label.strip())
        return normalized.strip("_")

    def _select_best_result(self, round_results, best_metric):
        def metric_value(result):
            if best_metric == "roc_auc":
                return result.get("combined_score", 0)
            return result.get("combined_score", 0)

        return max(round_results, key=metric_value)
