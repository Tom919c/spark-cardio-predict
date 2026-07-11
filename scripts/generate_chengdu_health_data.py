"""Generate reproducible synthetic population health records for dashboard analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROWS = 200_000
DEFAULT_SEED = 20260711
DEFAULT_OUTPUT = Path("data/dws/chengdu_resident_health_simulated.csv")

DISTRICTS = [
    "Jinjiang",
    "Qingyang",
    "Jinniu",
    "Wuhou",
    "Chenghua",
    "Longquanyi",
    "Qingbaijiang",
    "Xindu",
    "Wenjiang",
    "Shuangliu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def generate_dataset(rows: int, seed: int, output: Path) -> dict:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 91, size=rows)
    gender = rng.integers(0, 2, size=rows)
    bmi = np.round(rng.normal(24.8, 4.8, size=rows).clip(16.0, 42.0), 2)
    cholesterol = rng.choice([1, 2, 3], size=rows, p=[0.62, 0.27, 0.11])
    diabetes = rng.binomial(1, 0.16, size=rows)
    hypertension = rng.binomial(1, 0.34, size=rows)
    smoker = rng.choice([0, 1, 2], size=rows, p=[0.56, 0.16, 0.28])
    alcohol = rng.binomial(1, 0.43, size=rows)
    exercise = rng.binomial(1, 0.25, size=rows)
    district = rng.choice(DISTRICTS, size=rows)
    age_group = pd.cut(
        age,
        bins=[17, 39, 59, 74, 90],
        labels=["18-39", "40-59", "60-74", "75-90"],
    ).astype(str)

    heart_score = (
        0.02 * (age - 40)
        + 0.35 * hypertension
        + 0.25 * diabetes
        + 0.18 * (cholesterol >= 2)
        + 0.14 * (smoker == 2)
        + 0.12 * (bmi >= 28)
        - 0.16 * exercise
        + rng.normal(0, 0.55, size=rows)
    )
    stroke_score = (
        0.023 * (age - 45)
        + 0.42 * hypertension
        + 0.18 * diabetes
        + 0.17 * (cholesterol >= 2)
        + 0.10 * (smoker > 0)
        - 0.12 * exercise
        + rng.normal(0, 0.58, size=rows)
    )
    label_heart = (heart_score >= np.quantile(heart_score, 0.88)).astype(int)
    label_stroke = (stroke_score >= np.quantile(stroke_score, 0.975)).astype(int)
    target_disease = np.where(label_stroke == 1, 2, np.where(label_heart == 1, 1, 0))

    frame = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "cholesterol": cholesterol,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "smoker": smoker,
            "alcohol": alcohol,
            "exercise": exercise,
            "target_disease": target_disease,
            "region": "Chengdu",
            "district": district,
            "age_group": age_group,
            "label_heart": label_heart,
            "label_stroke": label_stroke,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    profile = {
        "dataset": str(output).replace("\\", "/"),
        "rows": int(len(frame)),
        "seed": seed,
        "heart_rate": round(float(frame["label_heart"].mean()), 4),
        "stroke_rate": round(float(frame["label_stroke"].mean()), 4),
    }
    output.with_name(f"{output.stem}_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def main() -> None:
    args = parse_args()
    profile = generate_dataset(args.rows, args.seed, args.output)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
