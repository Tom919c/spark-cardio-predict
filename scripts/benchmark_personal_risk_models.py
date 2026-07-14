"""Compare calibrated dual-risk candidates on natural-prevalence data.

The script deliberately keeps model selection separate from the active-model
registry.  It reports discrimination, calibration, decision thresholds, and
fixed clinical stress scenarios before any model is promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BaseConfig
from src.ml.calibrated_model import CalibratedRiskModel
from src.ml.evaluate import ModelEvaluator
from src.services.data_service import DataService

try:
    from xgboost import XGBClassifier
except ImportError:  # Optional until the dependency is installed for comparison.
    XGBClassifier = None


SCENARIOS = {
    "healthy_35": [35, 0, 21, 1, 0, 0, 0, 0, 1],
    "moderate_55": [55, 1, 28, 2, 0, 1, 1, 1, 0],
    "severe_35": [35, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_45": [45, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_55": [55, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_65": [65, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_75": [75, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_85": [85, 1, 35, 3, 1, 1, 2, 1, 0],
}
MONOTONIC_CONSTRAINTS = [1, 0, 1, 1, 1, 1, 1, 1, -1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-limit", type=int, default=250_000)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def candidate_factories(random_state):
    common_hgb = {
        "learning_rate": 0.06,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": random_state,
    }
    factories = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(**common_hgb),
        "hist_gradient_boosting_monotonic": lambda: HistGradientBoostingClassifier(
            **common_hgb, monotonic_cst=MONOTONIC_CONSTRAINTS
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=320,
            max_depth=16,
            min_samples_leaf=4,
            max_features=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest_tuned": lambda: RandomForestClassifier(
            n_estimators=320,
            max_depth=16,
            min_samples_leaf=5,
            max_features=0.8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        xgb_common = {
            "n_estimators": 360,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "min_child_weight": 6,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
        }
        factories["xgboost"] = lambda: XGBClassifier(**xgb_common)
        factories["xgboost_monotonic"] = lambda: XGBClassifier(
            **xgb_common,
            monotone_constraints=tuple(MONOTONIC_CONSTRAINTS),
        )
    return factories


def select_threshold(y_true, probabilities, target_name):
    """Choose from calibration data, prioritising recall without hiding precision."""
    required_recall = 0.55 if target_name == "heart" else 0.70
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = [
        (float(threshold), float(p), float(r))
        for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds)
        if r >= required_recall
    ]
    if candidates:
        return max(candidates, key=lambda item: (item[1], item[0]))[0]
    return 0.5


def evaluate_target(features, target, target_name, factories, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state, stratify=target
    )
    x_fit, x_calibrate, y_fit, y_calibrate = train_test_split(
        x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    scenario_frame = pd.DataFrame(SCENARIOS.values(), columns=features.columns)
    evaluator = ModelEvaluator()
    results = {}

    for name, factory in factories.items():
        started = time.perf_counter()
        estimator = factory()
        estimator.fit(x_fit, y_fit)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(estimator.predict_proba(x_calibrate)[:, 1], y_calibrate)
        model = CalibratedRiskModel(estimator, calibrator, features.columns)
        probabilities = model.predict_proba(x_test)[:, 1]
        threshold = select_threshold(y_calibrate, model.predict_proba(x_calibrate)[:, 1], target_name)
        predictions = (probabilities >= threshold).astype(int)
        metrics = evaluator.evaluate_binary(y_test, predictions, probabilities)
        metrics["decision_threshold"] = round(float(threshold), 6)
        metrics["fit_seconds"] = round(time.perf_counter() - started, 2)
        metrics["scenarios"] = {
            scenario: round(float(probability), 6)
            for scenario, probability in zip(
                SCENARIOS, model.predict_proba(scenario_frame)[:, 1]
            )
        }
        results[name] = metrics
    return results


def main():
    args = parse_args()
    config = BaseConfig.as_dict()
    dataframe = DataService(config).get_training_dataframe()
    if len(dataframe) > args.sample_limit:
        dataframe = dataframe.sample(args.sample_limit, random_state=config["RANDOM_STATE"])
    features = dataframe.loc[:, config["CARDIO_FEATURE_COLUMNS"]]
    factories = candidate_factories(config["RANDOM_STATE"])
    result = {
        "sample_rows": int(len(dataframe)),
        "natural_prevalence_test": True,
        "strategy": "fit_calibration_test_split_with_isotonic_calibration",
        "targets": {
            name: evaluate_target(
                features, dataframe[column].astype(int), name, factories, config["RANDOM_STATE"]
            )
            for name, column in config["RISK_TARGETS"].items()
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
