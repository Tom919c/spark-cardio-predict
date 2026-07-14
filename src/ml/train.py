"""双独立 XGBoost 训练：概率校准、阈值优化和训练耗时估算。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.ml.calibrated_model import CalibratedRiskModel
from src.ml.evaluate import ModelEvaluator


class CardioModelTrainer:
    """训练心脏事件与脑卒中两个独立 XGBoost 模型。"""

    def __init__(self, config, progress_callback=None):
        self.config = config
        self.feature_columns = config["CARDIO_FEATURE_COLUMNS"]
        self.model_output_dir = Path(config["MODEL_OUTPUT_DIR"])
        self.random_state = config["RANDOM_STATE"]
        self.test_size = config["TRAIN_TEST_SPLIT_RATIO"]
        self.n_estimators = config.get("XGBOOST_TREES", 360)
        self.targets = config["RISK_TARGETS"]
        self.stroke_min_recall = config.get("STROKE_MIN_RECALL", 0.70)
        self.progress_callback = progress_callback or print
        self._started_at = time.perf_counter()

    def _progress(self, message):
        elapsed = time.perf_counter() - self._started_at
        self.progress_callback(f"[训练进度 {elapsed:.1f}s] {message}")

    def estimate(self, dataframe, benchmark_trees=80):
        """用小规模 XGBoost 估算本次双模型训练耗时，不写入模型文件。"""
        started = time.perf_counter()
        sample = dataframe.sample(min(len(dataframe), 30_000), random_state=self.random_state)
        estimates = {}
        for target_name, target_column in self.targets.items():
            features = sample[self.feature_columns]
            target = sample[target_column].astype(int)
            x_train, _, y_train, _ = train_test_split(
                features, target, test_size=self.test_size,
                random_state=self.random_state, stratify=target,
            )
            benchmark = self._build_estimator(n_estimators=benchmark_trees)
            fit_started = time.perf_counter()
            benchmark.fit(x_train, y_train)
            benchmark_seconds = time.perf_counter() - fit_started
            scale = max(len(dataframe) / len(sample), 1.0) * self.n_estimators / benchmark_trees
            estimates[target_name] = round(benchmark_seconds * scale, 1)
        total = round(sum(estimates.values()), 1)
        return {
            "sample_rows": len(sample),
            "benchmark_trees": benchmark_trees,
            "estimated_seconds": total,
            "estimated_minutes": round(total / 60, 1),
            "per_model_seconds": estimates,
            "message": f"预计双模型训练耗时约 {round(total / 60, 1)} 分钟，实际受 CPU、内存和数据模式影响。",
            "benchmark_seconds": round(time.perf_counter() - started, 1),
        }

    def train(self, dataframe, run_label="phase1"):
        """训练两个模型，并返回指标、阈值、耗时和数据分布信息。"""
        self._started_at = time.perf_counter()
        estimate = self.estimate(dataframe)
        self._progress(estimate["message"])
        results = {}
        total_started = time.perf_counter()
        for index, (target_name, target_column) in enumerate(self.targets.items(), 1):
            results[target_name] = self._train_target(
                dataframe, target_name, target_column, run_label, index, len(self.targets)
            )
        elapsed = round(time.perf_counter() - total_started, 1)
        self._progress(f"全部模型完成，实际训练耗时 {elapsed:.1f}s。")
        return {
            "strategy": "xgboost_with_natural_prevalence_training_isotonic_calibration_and_threshold_tuning",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": self.feature_columns,
            "training_estimate": estimate,
            "training_seconds": elapsed,
            "targets": results,
        }

    def _train_target(self, dataframe, target_name, target_column, run_label, index, total):
        target_started = time.perf_counter()
        features = dataframe.loc[:, self.feature_columns]
        target = dataframe[target_column].astype(int)
        weights = dataframe.get("sample_weight")
        x_train, x_test, y_train, y_test, weight_train, _ = train_test_split(
            features, target, weights, test_size=self.test_size,
            random_state=self.random_state, stratify=target,
        )
        x_fit, x_calibrate, y_fit, y_calibrate, weight_fit, weight_calibrate = train_test_split(
            x_train, y_train, weight_train, test_size=0.2,
            random_state=self.random_state, stratify=y_train,
        )

        self._progress(
            f"开始 {target_name} 模型（{index}/{total}），拟合样本 {len(x_fit):,}。"
        )
        estimator = self._build_estimator()
        estimator.fit(x_fit, y_fit, sample_weight=weight_fit)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            estimator.predict_proba(x_calibrate)[:, 1], y_calibrate,
            sample_weight=weight_calibrate,
        )
        calibrated_probabilities = calibrator.predict(estimator.predict_proba(x_calibrate)[:, 1])
        threshold = self._select_threshold(y_calibrate, calibrated_probabilities, target_name)
        model = CalibratedRiskModel(estimator, calibrator, self.feature_columns, threshold)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        metrics = ModelEvaluator().evaluate_binary(y_test, predictions, probabilities)
        metrics["decision_threshold"] = round(float(threshold), 6)
        metrics["train_positive_rate"] = round(float(y_fit.mean()), 6)
        metrics["test_positive_rate"] = round(float(y_test.mean()), 6)
        metrics["oversampled_rows"] = 0
        model_path = self._save_model(model, target_name, run_label)
        elapsed = round(time.perf_counter() - target_started, 1)
        self._progress(f"{target_name} 模型完成，耗时 {elapsed:.1f}s，阈值 {threshold:.4f}。")
        return {
            "target_column": target_column,
            "model_path": str(model_path),
            "decision_threshold": threshold,
            "training_seconds": elapsed,
            "metrics": metrics,
            "feature_importance": dict(zip(self.feature_columns, map(float, model.feature_importances_))),
        }

    def _select_threshold(self, y_true, probabilities, target_name):
        required_recall = 0.55 if target_name == "heart" else self.stroke_min_recall
        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        candidates = [
            (float(threshold), float(p), float(r))
            for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds)
            if r >= required_recall
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]
        return 0.5

    def _build_estimator(self, n_estimators=None):
        """Return the selected production estimator with deterministic settings."""
        return XGBClassifier(
            n_estimators=n_estimators or self.n_estimators,
            max_depth=self.config.get("XGBOOST_MAX_DEPTH", 5),
            learning_rate=self.config.get("XGBOOST_LEARNING_RATE", 0.05),
            subsample=self.config.get("XGBOOST_SUBSAMPLE", 0.85),
            colsample_bytree=self.config.get("XGBOOST_COLSAMPLE_BYTREE", 0.90),
            min_child_weight=self.config.get("XGBOOST_MIN_CHILD_WEIGHT", 6.0),
            gamma=self.config.get("XGBOOST_GAMMA", 0.0),
            reg_alpha=self.config.get("XGBOOST_REG_ALPHA", 0.0),
            reg_lambda=self.config.get("XGBOOST_REG_LAMBDA", 2.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _save_model(self, model, target_name, run_label):
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c for c in run_label if c.isalnum() or c in "_-") or "run"
        path = self.model_output_dir / f"xgboost_{target_name}_{safe_label}_{timestamp}.joblib"
        joblib.dump(model, path)
        return path
