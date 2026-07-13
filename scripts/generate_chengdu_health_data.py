"""Generate reproducible synthetic Chengdu resident health records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROWS = 2_000_000
DEFAULT_SEED = 20260711
DEFAULT_OUTPUT = Path("data/raw/chengdu_resident_health_simulated.csv")

DISTRICTS = [
    "锦江区",
    "青羊区",
    "金牛区",
    "武侯区",
    "成华区",
    "龙泉驿区",
    "青白江区",
    "新都区",
    "温江区",
    "双流区",
    "郫都区",
    "新津区",
    "都江堰市",
    "彭州市",
    "邛崃市",
    "崇州市",
    "简阳市",
    "金堂县",
    "大邑县",
    "蒲江县",
]

DISTRICT_PROBABILITIES = np.array(
    [0.08, 0.08, 0.09, 0.11, 0.10, 0.08, 0.04, 0.09, 0.08, 0.10,
     0.09, 0.03, 0.03, 0.03, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01],
    dtype=float,
)
DISTRICT_PROBABILITIES /= DISTRICT_PROBABILITIES.sum()

# The values are taken from mydocs/成都市居民健康状况.md.
BASE_RATES = {
    "hypertension": 0.3355,
    "diabetes": 0.1597,
    "cholesterol_abnormal": 0.3557,
    "current_smoker": 0.2774,
    "overweight": 0.3602,
    "obesity": 0.1539,
    "exercise": 0.2540,
    "alcohol": 0.4307,
    "secondhand_smoke": 0.6098,
    "annual_stroke_event": 500.28 / 100_000,
    "annual_heart_attack_event": 79.36 / 100_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def sample_binary_by_score(score: np.ndarray, prevalence: float) -> np.ndarray:
    """Select a fixed prevalence while retaining correlations in the score."""
    cutoff = np.quantile(score, 1.0 - prevalence)
    return (score >= cutoff).astype(np.int8)


def generate_chunk(rng: np.random.Generator, start: int, size: int) -> pd.DataFrame:
    age = np.clip(np.rint(rng.normal(57.0, 16.0, size)), 18, 90).astype(np.int16)
    gender = rng.binomial(1, 0.49, size).astype(np.int8)

    # Generate BMI categories directly so the city-level overweight and obesity
    # rates remain close to the documented baseline.
    bmi_category = rng.choice(3, size=size, p=[0.6398, 0.2063, 0.1539])
    bmi = np.empty(size, dtype=np.float32)
    normal = bmi_category == 0
    overweight = bmi_category == 1
    obese = bmi_category == 2
    bmi[normal] = np.clip(rng.normal(21.8, 1.7, normal.sum()), 16.0, 24.9)
    bmi[overweight] = np.clip(rng.normal(27.0, 1.3, overweight.sum()), 25.0, 29.9)
    bmi[obese] = np.clip(rng.normal(32.5, 2.6, obese.sum()), 30.0, 45.0)
    bmi = np.round(bmi, 2)

    height_cm = np.where(
        gender == 1,
        rng.normal(171.5, 6.2, size),
        rng.normal(160.0, 5.8, size),
    )
    height_cm = np.round(np.clip(height_cm, 145.0, 195.0), 1)
    weight_kg = np.round(bmi * (height_cm / 100.0) ** 2, 1)

    age_factor = (age - 45) / 18.0
    exercise_score = 0.9 - 0.35 * (bmi >= 25) - 0.16 * np.maximum(age - 60, 0) / 30
    exercise_score += rng.normal(0, 0.55, size)
    exercise = sample_binary_by_score(exercise_score, BASE_RATES["exercise"])

    smoker_score = 0.35 + 0.35 * (gender == 1) + 0.10 * (age < 60)
    smoker_score += rng.normal(0, 0.45, size)
    current_smoker = sample_binary_by_score(
        smoker_score, BASE_RATES["current_smoker"]
    )
    former_score = 0.20 + 0.40 * (age >= 55) + 0.20 * (gender == 1)
    former_score += rng.normal(0, 0.55, size)
    former_smoker = (current_smoker == 0) & (former_score >= np.quantile(former_score, 0.78))
    smoker = np.where(current_smoker == 1, 2, np.where(former_smoker, 1, 0)).astype(np.int8)

    alcohol_score = 0.35 + 0.35 * (gender == 1) + 0.10 * (smoker > 0)
    alcohol_score += rng.normal(0, 0.50, size)
    alcohol = sample_binary_by_score(alcohol_score, BASE_RATES["alcohol"])

    salt_intake_g_day = np.where(
        rng.random(size) < 0.65,
        rng.uniform(8.0, 12.5, size),
        rng.uniform(4.0, 8.0, size),
    )
    salt_intake_g_day = np.round(salt_intake_g_day, 2)

    # Chronic-condition scores share age, BMI, diet, and exercise signals.
    diabetes_score = 0.72 * age_factor + 0.55 * (bmi >= 25) + 0.25 * (bmi >= 30)
    diabetes_score += 0.16 * (exercise == 0) + rng.normal(0, 0.80, size)
    diabetes = sample_binary_by_score(diabetes_score, BASE_RATES["diabetes"])

    hypertension_score = (
        0.85 * age_factor
        + 0.55 * (bmi >= 25)
        + 0.45 * (bmi >= 30)
        + 0.24 * (salt_intake_g_day > 8)
        + 0.20 * diabetes
        - 0.35 * exercise
        + rng.normal(0, 0.75, size)
    )
    hypertension_score += 0.18 * diabetes
    hypertension = sample_binary_by_score(hypertension_score, BASE_RATES["hypertension"])

    cholesterol_score = (
        0.42 * age_factor
        + 0.30 * (bmi >= 25)
        + 0.18 * (exercise == 0)
        + 0.14 * diabetes
        + rng.normal(0, 0.75, size)
    )
    abnormal_cholesterol = sample_binary_by_score(
        cholesterol_score, BASE_RATES["cholesterol_abnormal"]
    )
    high_cholesterol = abnormal_cholesterol & (
        cholesterol_score >= np.quantile(cholesterol_score, 0.87)
    )
    cholesterol = np.where(high_cholesterol, 3, np.where(abnormal_cholesterol, 2, 1)).astype(np.int8)

    systolic_bp = 116 + 0.62 * age + 3.8 * hypertension + 2.4 * (bmi >= 30)
    systolic_bp += rng.normal(0, 10, size)
    systolic_bp = np.round(np.clip(systolic_bp, 90, 220)).astype(np.int16)
    diastolic_bp = 70 + 0.20 * (age - 40) + 4.5 * hypertension + rng.normal(0, 7, size)
    diastolic_bp = np.round(np.clip(diastolic_bp, 50, 130)).astype(np.int16)

    secondhand_score = 0.45 + 0.20 * (gender == 1) + 0.20 * (age < 55) + rng.normal(0, 0.65, size)
    secondhand_smoke = np.where(
        smoker == 0,
        sample_binary_by_score(secondhand_score, BASE_RATES["secondhand_smoke"]),
        0,
    ).astype(np.int8)
    fruit_veg_insufficient = (
        (rng.random(size) < 0.422) | ((salt_intake_g_day > 9.5) & (exercise == 0))
    ).astype(np.int8)

    risk_score = (
        0.95 * age_factor
        + 0.58 * hypertension
        + 0.48 * diabetes
        + 0.34 * (cholesterol >= 2)
        + 0.28 * (smoker == 2)
        + 0.18 * (smoker == 1)
        + 0.22 * alcohol
        + 0.24 * (bmi >= 30)
        + 0.18 * (salt_intake_g_day > 8)
        - 0.38 * exercise
        + rng.normal(0, 0.80, size)
    )
    heart_risk_score = risk_score + 0.25 * (gender == 1) + rng.normal(0, 0.35, size)
    stroke_risk_score = risk_score + 0.25 * hypertension + 0.18 * (salt_intake_g_day > 8)
    stroke_risk_score += rng.normal(0, 0.45, size)

    label_chd = sample_binary_by_score(heart_risk_score, 0.115)
    label_heart_attack = sample_binary_by_score(heart_risk_score + rng.normal(0, 0.9, size), 0.010)
    label_heart_failure = sample_binary_by_score(heart_risk_score + 0.25 * age_factor + rng.normal(0, 0.9, size), 0.015)
    label_heart = ((label_chd == 1) | (label_heart_attack == 1) | (label_heart_failure == 1)).astype(np.int8)
    label_stroke = sample_binary_by_score(stroke_risk_score, 0.025)

    # target_disease preserves the existing project's 0/1/2 contract. The
    # independent labels above retain co-morbidity information for the newer design.
    target_disease = np.where(label_stroke == 1, 2, np.where(label_heart == 1, 1, 0)).astype(np.int8)

    annual_stroke_event = (
        rng.random(size) < BASE_RATES["annual_stroke_event"] * np.clip(1 + 1.5 * (stroke_risk_score > np.median(stroke_risk_score)), 0.5, 3.0)
    ).astype(np.int8)
    annual_heart_attack_event = (
        rng.random(size) < BASE_RATES["annual_heart_attack_event"] * np.clip(1 + 1.8 * (heart_risk_score > np.median(heart_risk_score)), 0.5, 3.0)
    ).astype(np.int8)

    district = rng.choice(DISTRICTS, size=size, p=DISTRICT_PROBABILITIES)
    district_code = pd.Categorical(district, categories=DISTRICTS).codes.astype(np.int16) + 1
    community_number = rng.integers(1, 101, size=size, dtype=np.int16)
    community_id = district_code * 1000 + community_number
    age_group = pd.cut(
        age,
        bins=[17, 39, 59, 74, 90],
        labels=["18-39", "40-59", "60-74", "75-90"],
    ).astype(str)

    return pd.DataFrame(
        {
            # Keep the model-compatible columns first for easy inspection and ETL use.
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
            "resident_id": np.arange(start + 510100000001, start + size + 510100000001, dtype=np.int64),
            "region": "成都市",
            "district": district,
            "community_id": community_id,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "salt_intake_g_day": salt_intake_g_day,
            "fruit_veg_insufficient": fruit_veg_insufficient,
            "secondhand_smoke": secondhand_smoke,
            "age_group": age_group,
            "label_heart": label_heart,
            "label_stroke": label_stroke,
            "label_chd": label_chd,
            "label_heart_attack": label_heart_attack,
            "label_heart_failure": label_heart_failure,
            "annual_stroke_event": annual_stroke_event,
            "annual_heart_attack_event": annual_heart_attack_event,
        }
    )


def generate_dataset(rows: int, seed: int, output: Path, chunk_size: int) -> dict:
    if rows <= 0:
        raise ValueError("rows must be greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk-size must be greater than 0")

    output.parent.mkdir(parents=True, exist_ok=True)
    profile_path = output.with_name(f"{output.stem}_profile.json")
    if output.exists():
        output.unlink()

    rng = np.random.default_rng(seed)
    written = 0
    first_chunk = True
    for start in range(0, rows, chunk_size):
        size = min(chunk_size, rows - start)
        frame = generate_chunk(rng, start, size)
        frame.to_csv(
            output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            float_format="%.2f",
        )
        first_chunk = False
        written += size
        print(f"generated {written:,}/{rows:,} rows")

    validation = pd.read_csv(output)
    profile = {
        "dataset": str(output).replace("\\", "/"),
        "rows": int(len(validation)),
        "columns": validation.columns.tolist(),
        "seed": seed,
        "source_document": "mydocs/成都居民健康数据.md",
        "base_rates": BASE_RATES,
        "observed_rates": {
            "hypertension": round(float(validation["hypertension"].mean()), 6),
            "diabetes": round(float(validation["diabetes"].mean()), 6),
            "cholesterol_abnormal": round(float((validation["cholesterol"] >= 2).mean()), 6),
            "current_smoker": round(float((validation["smoker"] == 2).mean()), 6),
            "overweight": round(float((validation["bmi"] >= 25).mean()), 6),
            "obesity": round(float((validation["bmi"] >= 30).mean()), 6),
            "exercise": round(float(validation["exercise"].mean()), 6),
            "alcohol": round(float(validation["alcohol"].mean()), 6),
            "secondhand_smoke_all": round(float(validation["secondhand_smoke"].mean()), 6),
            "target_disease_0": round(float((validation["target_disease"] == 0).mean()), 6),
            "target_disease_1": round(float((validation["target_disease"] == 1).mean()), 6),
            "target_disease_2": round(float((validation["target_disease"] == 2).mean()), 6),
        },
        "compatibility": {
            "model_feature_columns": [
                "age", "gender", "bmi", "cholesterol", "diabetes",
                "hypertension", "smoker", "alcohol", "exercise",
            ],
            "legacy_target_column": "target_disease",
            "extended_labels_preserve_comorbidity": True,
        },
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def main() -> None:
    args = parse_args()
    profile = generate_dataset(args.rows, args.seed, args.output, args.chunk_size)
    print(json.dumps(profile["observed_rates"], ensure_ascii=False, indent=2))
    print(f"profile written to {args.output.with_name(args.output.stem + '_profile.json')}")


if __name__ == "__main__":
    main()
