"""Two independent random-forest training with probability calibration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.ml.evaluate import ModelEvaluator
from src.ml.calibrated_model import CalibratedRiskModel


class CardioModelTrainer:
    """Trains dual random forest models for heart-risk and stroke-risk prediction."""

    def __init__(self, config):
        self.config = config
        self.feature_columns = config["CARDIO_FEATURE_COLUMNS"]
        self.model_output_dir = Path(config["MODEL_OUTPUT_DIR"])
        self.random_state = config["RANDOM_STATE"]
        self.test_size = config["TRAIN_TEST_SPLIT_RATIO"]
        self.n_estimators = config["RANDOM_FOREST_TREES"]
        self.targets = config["RISK_TARGETS"]

    def train(self, dataframe, run_label="phase1"):
        results = {}
        for target_name, target_column in self.targets.items():
            results[target_name] = self._train_target(
                dataframe=dataframe,
                target_name=target_name,
                target_column=target_column,
                run_label=run_label,
            )
        return {
            "strategy": "two_independent_random_forests_with_isotonic_calibration",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": self.feature_columns,
            "targets": results,
        }

    def _train_target(self, dataframe, target_name, target_column, run_label):
        features = dataframe[self.feature_columns].copy()
        target = dataframe[target_column].astype(int)
        weights = dataframe["sample_weight"] if "sample_weight" in dataframe else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )
        weight_train = weights.loc[x_train.index] if weights is not None else None
        estimator = self._build_model()
        estimator.fit(x_train, y_train, sample_weight=weight_train)
        raw_train_probability = estimator.predict_proba(x_train)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_train_probability, y_train)
        model = CalibratedRiskModel(estimator, calibrator, self.feature_columns)

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_binary(y_test, predictions, probabilities)
        run_id = self._build_run_id(target_name=target_name, run_label=run_label)
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

    def _build_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            random_state=self.random_state,
            class_weight="balanced_subsample",
        )

    def _build_run_id(self, target_name, run_label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_label = (run_label or "phase1").strip().replace(" ", "_")
        return f"random_forest_{target_name}_{normalized_label}_{timestamp}"
