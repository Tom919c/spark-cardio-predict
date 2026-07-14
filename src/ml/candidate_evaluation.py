"""阶段二候选模型比较，不写入 active 模型清单。"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluate import ModelEvaluator


class CandidateModelEvaluator:
    """分别评估心脏和卒中候选模型，保留自然患病率测试集。"""

    def __init__(self, config):
        self.config = config
        self.features = list(config.get("CARDIO_FEATURE_COLUMNS", []))
        self.targets = dict(config.get("RISK_TARGETS", {"heart": "label_heart", "stroke": "label_stroke"}))
        self.random_state = int(config.get("RANDOM_STATE", 42))
        self.evaluator = ModelEvaluator()

    def evaluate(self, dataframe, include_stacking=False, sample_limit=150_000):
        missing = [field for field in self.features if field not in dataframe]
        if missing:
            raise ValueError(f"候选模型评估缺少特征列: {missing}")
        working = dataframe.copy()
        if len(working) > sample_limit:
            working = working.sample(sample_limit, random_state=self.random_state)
        features = working[self.features].apply(pd.to_numeric, errors="coerce").fillna(0)
        result = {
            "candidate_only": True,
            "sample_rows": int(len(working)),
            "natural_prevalence_test": True,
            "targets": {},
        }
        for name, target in self.targets.items():
            if target not in working:
                continue
            result["targets"][name] = self._evaluate_target(
                features, working[target].astype(int), include_stacking
            )
        return result

    def _evaluate_target(self, features, target, include_stacking):
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=float(self.config.get("TRAIN_TEST_SPLIT_RATIO", 0.2)),
            random_state=self.random_state,
            stratify=target,
        )
        estimators = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced", random_state=self.random_state),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=min(int(self.config.get("RANDOM_FOREST_TREES", 200)), 200),
                max_depth=10,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=31,
                random_state=self.random_state,
            ),
        }
        if include_stacking:
            estimators["stacking"] = StackingClassifier(
                estimators=[
                    ("lr", estimators["logistic_regression"]),
                    ("rf", estimators["random_forest"]),
                ],
                final_estimator=LogisticRegression(max_iter=200),
                cv=3,
                stack_method="predict_proba",
                n_jobs=-1,
            )

        results = {}
        for name, estimator in estimators.items():
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            probabilities = estimator.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = self.evaluator.evaluate_binary(y_test, predictions, probabilities)
            metrics["fit_seconds"] = round(time.perf_counter() - started, 2)
            results[name] = metrics
        return results
