"""Search XGBoost configurations without changing the active model registry.

The search uses three disjoint partitions: fit, selection, and holdout.  The
holdout partition is only used after configuration selection.  The top
configurations are then refit with several seeds to check stability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BaseConfig


SCENARIOS = {
    "healthy_35": [35, 0, 21, 1, 0, 0, 0, 0, 1],
    "severe_35": [35, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_55": [55, 1, 35, 3, 1, 1, 2, 1, 0],
    "severe_75": [75, 1, 35, 3, 1, 1, 2, 1, 0],
}

CONFIGURATIONS = [
    {
        "name": "current_baseline",
        "n_estimators": 360, "max_depth": 5, "learning_rate": 0.05,
        "min_child_weight": 6, "subsample": 0.85, "colsample_bytree": 0.90,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0,
    },
    {
        "name": "depth4_low_lr",
        "n_estimators": 650, "max_depth": 4, "learning_rate": 0.03,
        "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0,
    },
    {
        "name": "depth5_low_lr",
        "n_estimators": 700, "max_depth": 5, "learning_rate": 0.03,
        "min_child_weight": 4, "subsample": 0.90, "colsample_bytree": 0.90,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 3.0,
    },
    {
        "name": "depth6_regularized",
        "n_estimators": 520, "max_depth": 6, "learning_rate": 0.04,
        "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.90,
        "gamma": 0.05, "reg_alpha": 0.0, "reg_lambda": 4.0,
    },
    {
        "name": "depth3_smooth",
        "n_estimators": 850, "max_depth": 3, "learning_rate": 0.025,
        "min_child_weight": 3, "subsample": 0.95, "colsample_bytree": 1.00,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0,
    },
    {
        "name": "depth4_high_min_child",
        "n_estimators": 600, "max_depth": 4, "learning_rate": 0.04,
        "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85,
        "gamma": 0.05, "reg_alpha": 0.05, "reg_lambda": 4.0,
    },
    {
        "name": "depth5_l1_regularized",
        "n_estimators": 500, "max_depth": 5, "learning_rate": 0.045,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.90,
        "gamma": 0.05, "reg_alpha": 0.20, "reg_lambda": 3.0,
    },
    {
        "name": "depth5_more_subsample",
        "n_estimators": 550, "max_depth": 5, "learning_rate": 0.04,
        "min_child_weight": 5, "subsample": 0.75, "colsample_bytree": 0.80,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 3.0,
    },
    {
        "name": "depth6_low_lr",
        "n_estimators": 800, "max_depth": 6, "learning_rate": 0.025,
        "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.85,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 4.0,
    },
    {
        "name": "depth4_fast",
        "n_estimators": 380, "max_depth": 4, "learning_rate": 0.07,
        "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.90,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/CVD_Standard_DWD_refined.csv")
    parser.add_argument("--output", default="data/xgboost_hyperparameter_search.json")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", default="42,2027,7")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def estimator(config, seed, target_name):
    # A small positive-class weight is tested for both targets without
    # replacing probability calibration; it can improve ranking for rare labels.
    scale_pos_weight = config.get("scale_pos_weight", 1.0)
    return XGBClassifier(
        **{key: value for key, value in config.items() if key != "name" and key != "scale_pos_weight"},
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def threshold_for(y_true, probabilities, target_name):
    from sklearn.metrics import precision_recall_curve

    required_recall = 0.55 if target_name == "heart" else 0.70
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = [
        (float(t), float(p), float(r))
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if r >= required_recall
    ]
    return max(candidates, key=lambda item: item[1])[0] if candidates else 0.5


def evaluate(estimator_model, x_data, y_data, x_calibration, y_calibration, target_name):
    raw_calibration = estimator_model.predict_proba(x_calibration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_calibration, y_calibration)
    calibrated = calibrator.predict(estimator_model.predict_proba(x_data)[:, 1])
    threshold = threshold_for(y_calibration, calibrator.predict(raw_calibration), target_name)
    return {
        "roc_auc": round(float(roc_auc_score(y_data, calibrated)), 6),
        "pr_auc": round(float(average_precision_score(y_data, calibrated)), 6),
        "brier_score": round(float(brier_score_loss(y_data, calibrated)), 6),
        "decision_threshold": round(float(threshold), 6),
        "recall": round(float(((calibrated >= threshold) & (y_data == 1)).sum() / max((y_data == 1).sum(), 1)), 6),
        "calibrator": calibrator,
    }


def fit_one(config, seed, target_name, x_fit, y_fit, x_selection, y_selection, x_holdout, y_holdout):
    started = time.perf_counter()
    model = estimator(config, seed, target_name)
    model.fit(x_fit, y_fit, verbose=False)
    selection_raw = model.predict_proba(x_selection)[:, 1]
    selection_roc = roc_auc_score(y_selection, selection_raw)
    selection_pr = average_precision_score(y_selection, selection_raw)
    holdout = evaluate(model, x_holdout, y_holdout, x_selection, y_selection, target_name)
    holdout.pop("calibrator", None)
    return {
        "config": config["name"],
        "target": target_name,
        "seed": seed,
        "selection_roc_auc": round(float(selection_roc), 6),
        "selection_pr_auc": round(float(selection_pr), 6),
        "selection_score": round(float(0.65 * selection_pr + 0.35 * selection_roc), 6),
        "holdout": holdout,
        "fit_seconds": round(time.perf_counter() - started, 2),
    }


def scenario_values(config, seed, target_name, x_fit, y_fit, x_selection, y_selection):
    model = estimator(config, seed, target_name)
    model.fit(x_fit, y_fit, verbose=False)
    raw = model.predict_proba(x_selection)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_selection)
    frame = pd.DataFrame(SCENARIOS.values(), columns=x_fit.columns)
    return {
        name: round(float(value), 6)
        for name, value in zip(SCENARIOS, calibrator.predict(model.predict_proba(frame)[:, 1]))
    }


def main():
    args = parse_args()
    config = BaseConfig.as_dict()
    dataframe = pd.read_csv(args.input)
    if args.sample_limit and len(dataframe) > args.sample_limit:
        dataframe = dataframe.sample(args.sample_limit, random_state=args.seed)
    features = dataframe.loc[:, config["CARDIO_FEATURE_COLUMNS"]]
    split_features, holdout_features, split_targets, holdout_targets = train_test_split(
        features,
        dataframe.loc[:, list(config["RISK_TARGETS"].values())],
        test_size=0.20,
        random_state=args.seed,
        stratify=dataframe["label_stroke"],
    )
    fit_features, selection_features, fit_targets, selection_targets = train_test_split(
        split_features,
        split_targets,
        test_size=0.1875,  # 15% selection, 65% fit, 20% untouched holdout.
        random_state=args.seed,
        stratify=split_targets["label_stroke"],
    )

    results = {"heart": [], "stroke": []}
    for target_name, target_column in config["RISK_TARGETS"].items():
        for candidate in CONFIGURATIONS:
            results[target_name].append(
                fit_one(
                    candidate, args.seed, target_name,
                    fit_features, fit_targets[target_column],
                    selection_features, selection_targets[target_column],
                    holdout_features, holdout_targets[target_column],
                )
            )

    repeats = [int(value.strip()) for value in args.repeat_seeds.split(",") if value.strip()]
    stability = {"heart": [], "stroke": []}
    for target_name, target_column in config["RISK_TARGETS"].items():
        ranked = sorted(
            results[target_name],
            key=lambda item: item["selection_score"],
            reverse=True,
        )[: args.top_k]
        for item in ranked:
            candidate = next(c for c in CONFIGURATIONS if c["name"] == item["config"])
            for seed in repeats:
                stability[target_name].append(
                    fit_one(
                        candidate, seed, target_name,
                        fit_features, fit_targets[target_column],
                        selection_features, selection_targets[target_column],
                        holdout_features, holdout_targets[target_column],
                    )
                )

    selected = {}
    for target_name in ("heart", "stroke"):
        rows = [row for row in stability[target_name]]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["config"], []).append(row["holdout"])
        selected[target_name] = sorted(
            [
                {
                    "config": name,
                    "mean_pr_auc": round(float(np.mean([v["pr_auc"] for v in values])), 6),
                    "mean_roc_auc": round(float(np.mean([v["roc_auc"] for v in values])), 6),
                    "mean_brier_score": round(float(np.mean([v["brier_score"] for v in values])), 6),
                    "runs": len(values),
                }
                for name, values in grouped.items()
            ],
            key=lambda item: (item["mean_pr_auc"], item["mean_roc_auc"], -item["mean_brier_score"]),
            reverse=True,
        )

    output = {
        "dataset": str(args.input).replace("\\", "/"),
        "rows": int(len(dataframe)),
        "partition": {"fit": 0.65, "selection": 0.15, "holdout": 0.20},
        "configuration_count": len(CONFIGURATIONS),
        "repeat_seeds": repeats,
        "selection_rule": "0.65 * selection PR-AUC + 0.35 * selection ROC-AUC",
        "results": results,
        "stability_runs": stability,
        "selected_by_stability": selected,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["selected_by_stability"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
