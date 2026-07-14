"""Build a reproducible, risk-direction-consistent training dataset.

The input retains its demographic and health-factor columns.  Only the two
model labels are rebuilt from a documented stochastic screening-risk score so
that each modifiable risk factor has the expected direction.  This dataset is
for model development, not an epidemiological incidence estimate or diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "age", "gender", "bmi", "cholesterol", "diabetes", "hypertension",
    "smoker", "alcohol", "exercise",
]
DEFAULT_INPUT = Path("data/raw/CVD_Standard_DWD.csv")
DEFAULT_OUTPUT = Path("data/raw/CVD_Standard_DWD_refined.csv")
DEFAULT_SEED = 20260714

# These are screening-model positive-class rates, not annual incidence rates.
# Their purpose is to keep the simulated follow-up population at a reasonable
# scale while preserving enough positive samples for a stable student project.
TARGET_RATES = {"heart": 0.03, "stroke": 0.06}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _calibrate_intercept(score: np.ndarray, target_rate: float) -> float:
    lower, upper = -20.0, 20.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if float(_sigmoid(score + midpoint).mean()) < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _base_score(dataframe: pd.DataFrame) -> np.ndarray:
    age = dataframe["age"].to_numpy(dtype=float)
    bmi = dataframe["bmi"].to_numpy(dtype=float)
    cholesterol = dataframe["cholesterol"].to_numpy(dtype=int)
    diabetes = dataframe["diabetes"].to_numpy(dtype=int)
    hypertension = dataframe["hypertension"].to_numpy(dtype=int)
    smoker = dataframe["smoker"].to_numpy(dtype=int)
    alcohol = dataframe["alcohol"].to_numpy(dtype=int)
    exercise = dataframe["exercise"].to_numpy(dtype=int)
    gender = dataframe["gender"].to_numpy(dtype=int)

    score = (
        0.052 * np.maximum(age - 35, 0)
        + 0.11 * gender
        + 0.13 * np.maximum(bmi - 23, 0)
        + np.select([cholesterol >= 3, cholesterol >= 2], [0.76, 0.34], default=0)
        + 0.78 * diabetes
        + 0.96 * hypertension
        + np.select([smoker >= 2, smoker >= 1], [0.72, 0.26], default=0)
        + 0.08 * alcohol
        - 0.48 * exercise
        + 0.52 * ((diabetes == 1) & (hypertension == 1))
        + 0.31 * ((smoker >= 2) & (hypertension == 1))
        + 0.27 * ((bmi >= 30) & (hypertension == 1))
    )
    return score


def _draw_labels(
    dataframe: pd.DataFrame, rng: np.random.Generator, target_name: str
) -> tuple[np.ndarray, np.ndarray]:
    score = _base_score(dataframe)
    if target_name == "heart":
        score += 0.17 * dataframe["gender"].to_numpy(dtype=int)
        noise_scale = 1.20
    else:
        score += (
            0.22 * dataframe["hypertension"].to_numpy(dtype=int)
            + 0.16 * (dataframe["age"].to_numpy(dtype=float) >= 65)
        )
        noise_scale = 1.15
    score += rng.normal(0, noise_scale, len(dataframe))
    intercept = _calibrate_intercept(score, TARGET_RATES[target_name])
    probability = _sigmoid(score + intercept)
    return rng.binomial(1, probability).astype(np.int8), probability


def build_dataset(input_path: Path, output_path: Path, seed: int) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
    dataframe = pd.read_csv(input_path)
    missing = [column for column in FEATURES if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    rng = np.random.default_rng(seed)
    heart_label, heart_probability = _draw_labels(dataframe, rng, "heart")
    stroke_label, stroke_probability = _draw_labels(dataframe, rng, "stroke")
    dataframe["label_heart"] = heart_label
    dataframe["label_stroke"] = stroke_label
    dataframe["label_chd"] = heart_label
    dataframe["label_heart_attack"] = heart_label
    dataframe["label_heart_failure"] = 0
    dataframe["target_disease"] = np.where(
        stroke_label == 1, 2, np.where(heart_label == 1, 1, 0)
    ).astype(np.int8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, float_format="%.6f")

    age_bands = pd.cut(
        dataframe["age"], [0, 44, 54, 64, 74, 120],
        labels=["<=44", "45-54", "55-64", "65-74", "75+"],
    )
    profile = {
        "dataset": str(output_path).replace("\\", "/"),
        "source_dataset": str(input_path).replace("\\", "/"),
        "rows": int(len(dataframe)),
        "seed": seed,
        "label_semantics": (
            "基于多危险因素的筛查风险代理标签，仅用于模型训练；"
            "不等同于真实年度发病率、临床诊断或流行病学统计。"
        ),
        "target_positive_rates": TARGET_RATES,
        "observed_positive_rates": {
            "heart": round(float(heart_label.mean()), 6),
            "stroke": round(float(stroke_label.mean()), 6),
            "either": round(float(((heart_label == 1) | (stroke_label == 1)).mean()), 6),
        },
        "mean_latent_probability": {
            "heart": round(float(heart_probability.mean()), 6),
            "stroke": round(float(stroke_probability.mean()), 6),
        },
        "age_band_label_rates": {
            "heart": {
                str(key): round(float(value), 6)
                for key, value in dataframe.groupby(age_bands, observed=False)["label_heart"].mean().items()
            },
            "stroke": {
                str(key): round(float(value), 6)
                for key, value in dataframe.groupby(age_bands, observed=False)["label_stroke"].mean().items()
            },
        },
        "risk_direction_rules": {
            "increase": [
                "age", "bmi", "cholesterol", "diabetes", "hypertension",
                "current_smoking", "alcohol", "male_for_heart",
            ],
            "decrease": ["regular_exercise"],
        },
    }
    profile_path = output_path.with_name(f"{output_path.stem}_profile.json")
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def main() -> None:
    args = parse_args()
    profile = build_dataset(args.input, args.output, args.seed)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
