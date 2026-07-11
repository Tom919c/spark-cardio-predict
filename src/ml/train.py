"""双独立随机森林训练：校准集拆分 + 保序概率校准。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.ml.calibrated_model import CalibratedRiskModel
from src.ml.evaluate import ModelEvaluator


class CardioModelTrainer:
    """训练心脏事件与卒中事件两个独立随机森林模型。"""

    def __init__(self, config):
        self.config = config
        self.feature_columns = config["CARDIO_FEATURE_COLUMNS"]
        self.model_output_dir = Path(config["MODEL_OUTPUT_DIR"])
        self.random_state = config["RANDOM_STATE"]
        self.test_size = config["TRAIN_TEST_SPLIT_RATIO"]
        self.n_estimators = config["RANDOM_FOREST_TREES"]
        self.targets = config["RISK_TARGETS"]

    def train(self, dataframe, run_label="phase1"):
        """训练两个目标模型并返回产物路径、评估指标和特征重要性。"""
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
            "trained_at": datetime.utcnow().isoformat(),
            "features": self.feature_columns,
            "targets": results,
        }

    def _train_target(self, dataframe, target_name, target_column, run_label):
        features = dataframe.loc[:, self.feature_columns]
        target = dataframe[target_column].astype(int)
        weights = dataframe.get("sample_weight")

        # 拆出测试集，stratify 保证正负样本比例一致。
        x_train, x_test, y_train, y_test, weight_train, _ = train_test_split(
            features, target, weights,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

        # 从训练集中再拆出校准集，校准器不拟合训练样本，避免数据泄漏。
        x_fit, x_calibrate, y_fit, y_calibrate, weight_fit, weight_calibrate = train_test_split(
            x_train, y_train, weight_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train,
        )

        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        estimator.fit(x_fit, y_fit, sample_weight=weight_fit)

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            estimator.predict_proba(x_calibrate)[:, 1],
            y_calibrate,
            sample_weight=weight_calibrate,
        )
        model = CalibratedRiskModel(estimator, calibrator, self.feature_columns)

        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        metrics = ModelEvaluator().evaluate_binary(y_test, predictions, probabilities)

        model_path = self._save_model(model, target_name, run_label)
        return {
            "target_column": target_column,
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_importance": dict(
                zip(self.feature_columns, map(float, model.feature_importances_))
            ),
        }

    def _save_model(self, model, target_name, run_label):
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(
            c for c in run_label if c.isalnum() or c in "_-"
        ) or "run"
        filename = f"random_forest_{target_name}_{safe_label}_{timestamp}.joblib"
        path = self.model_output_dir / filename
        joblib.dump(model, path)
        return path
