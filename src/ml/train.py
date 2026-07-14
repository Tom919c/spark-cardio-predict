"""双独立 XGBoost 训练：不平衡处理、概率校准和训练耗时估算。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
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
        self.n_estimators = config["RANDOM_FOREST_TREES"]
        self.targets = config["RISK_TARGETS"]
        self.oversample_ratio = config.get("MINORITY_OVERSAMPLE_RATIO", 0.15)
        self.max_multiplier = config.get("MINORITY_MAX_MULTIPLIER", 3.0)
        self.stroke_min_recall = config.get("STROKE_MIN_RECALL", 0.70)
        self.max_depth = config.get("XGBOOST_MAX_DEPTH", 8)
        self.learning_rate = config.get("XGBOOST_LEARNING_RATE", 0.05)
        self.subsample = config.get("XGBOOST_SUBSAMPLE", 0.8)
        self.colsample_bytree = config.get("XGBOOST_COLSAMPLE_BYTREE", 0.8)
        self.device = config.get("XGBOOST_DEVICE", "cuda")
        self.progress_callback = progress_callback or print
        self._started_at = time.perf_counter()

    def _progress(self, message):
        elapsed = time.perf_counter() - self._started_at
        self.progress_callback(f"[训练进度 {elapsed:.1f}s] {message}")

    def estimate(self, dataframe, benchmark_trees=20):
        """用小规模 XGBoost 估算本次双模型训练耗时，不写入模型文件。"""
        started = time.perf_counter()
        sample = dataframe.sample(min(len(dataframe), 30_000), random_state=self.random_state)
        estimates = {}
        for target_name, target_column in self.targets.items():
            features = sample[self.feature_columns]
            target = sample[target_column].astype(int)
            x_train, _, y_train, _ = train_test_split(
                features,
                target,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=target,
            )
            benchmark = XGBClassifier(
                n_estimators=benchmark_trees,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                tree_method="hist",
                device=self.device,
                eval_metric="logloss",
                n_jobs=-1,
            )
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
            "device": self.device,
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
            "strategy": "xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": self.feature_columns,
            "training_estimate": estimate,
            "training_seconds": elapsed,
            "device": self.device,
            "targets": results,
        }

    def _train_target(self, dataframe, target_name, target_column, run_label, index, total):
        target_started = time.perf_counter()
        features = dataframe.loc[:, self.feature_columns]
        target = dataframe[target_column].astype(int)
        weights = dataframe.get("sample_weight")
        x_train, x_test, y_train, y_test, weight_train, _ = train_test_split(
            features,
            target,
            weights,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )
        x_fit, x_calibrate, y_fit, y_calibrate, weight_fit, weight_calibrate = train_test_split(
            x_train,
            y_train,
            weight_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train,
        )

        # 只增强拟合子集，校准集和测试集保持真实患病率，避免评估虚高。
        if target_name == "stroke":
            x_fit, y_fit, weight_fit, added = self._oversample_minority(
                x_fit, y_fit, weight_fit
            )
        else:
            added = 0

        self._progress(
            f"开始{target_name} 模型（{index}/{total}），拟合样本 {len(x_fit):,}，新增卒中样本 {added:,}。"
        )
        estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method="hist",
            device=self.device,
            eval_metric="logloss",
            n_jobs=-1,
        )
        estimator.fit(x_fit, y_fit, sample_weight=weight_fit)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            estimator.predict_proba(x_calibrate)[:, 1],
            y_calibrate,
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
        metrics["oversampled_rows"] = int(added)
        model_path = self._save_model(model, target_name, run_label)
        elapsed = round(time.perf_counter() - target_started, 1)
        self._progress(f"{target_name} 模型完成，耗时 {elapsed:.1f}s，阈值 {threshold:.4f}。")
        return {
            "target_column": target_column,
            "model_path": str(model_path),
            "decision_threshold": threshold,
            "training_seconds": elapsed,
            "device": self.device,
            "metrics": metrics,
            "feature_importance": dict(
                zip(self.feature_columns, map(float, model.feature_importances_))
            ),
        }

    def _oversample_minority(self, features, target, weights):
        positive_index = np.flatnonzero(target.to_numpy() == 1)
        negative_count = int((target == 0).sum())
        desired = min(
            int(negative_count * self.oversample_ratio),
            int(len(positive_index) * self.max_multiplier),
        )
        additional = max(desired - len(positive_index), 0)
        if additional == 0:
            return features, target, weights, 0
        rng = np.random.default_rng(self.random_state)
        chosen = rng.choice(positive_index, size=additional, replace=True)
        extra_features = features.iloc[chosen]
        extra_target = target.iloc[chosen]
        extra_weights = weights.iloc[chosen] if weights is not None else None
        output_features = pd.concat([features, extra_features], ignore_index=True)
        output_target = pd.concat(
            [target.reset_index(drop=True), extra_target.reset_index(drop=True)],
            ignore_index=True,
        )
        if weights is None:
            output_weights = None
        else:
            output_weights = pd.concat(
                [weights.reset_index(drop=True), extra_weights.reset_index(drop=True)],
                ignore_index=True,
            )
        return output_features, output_target, output_weights, additional

    def _select_threshold(self, y_true, probabilities, target_name):
        if target_name != "stroke":
            return 0.5
        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        candidates = [
            (float(threshold), float(p), float(r))
            for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds)
            if r >= self.stroke_min_recall
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]
        return 0.5

    def _save_model(self, model, target_name, run_label):
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c for c in run_label if c.isalnum() or c in "_-") or "run"
        path = self.model_output_dir / f"random_forest_{target_name}_{safe_label}_{timestamp}.joblib"
        joblib.dump(model, path)
        return path
