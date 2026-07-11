from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from src.ml.calibrated_model import CalibratedRiskModel
from src.ml.model_registry import ModelRegistry
from src.services.risk_service import RiskService


FEATURES = [
    "age",
    "gender",
    "bmi",
    "cholesterol",
    "diabetes",
    "hypertension",
    "smoker",
    "alcohol",
    "exercise",
]


def create_model(path: Path):
    features = pd.DataFrame(
        [[45, 0, 22, 1, 0, 0, 0, 0, 1], [72, 1, 32, 3, 1, 1, 2, 1, 0]],
        columns=FEATURES,
    )
    target = np.array([0, 1])
    estimator = RandomForestClassifier(n_estimators=5, random_state=42).fit(features, target)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.9], target)
    joblib.dump(CalibratedRiskModel(estimator, calibrator, FEATURES), path)


def test_registry_and_risk_service_use_server_managed_models(tmp_path: Path):
    heart_path = tmp_path / "random_forest_heart_test.joblib"
    stroke_path = tmp_path / "random_forest_stroke_test.joblib"
    create_model(heart_path)
    create_model(stroke_path)
    registry_path = tmp_path / "active_models.json"
    ModelRegistry(str(tmp_path), str(registry_path), FEATURES).register(
        {"heart": str(heart_path), "stroke": str(stroke_path)},
        {},
    )

    config = {
        "CARDIO_FEATURE_COLUMNS": FEATURES,
        "RISK_LEVEL_THRESHOLDS": (0.15, 0.35, 0.6, 0.8),
        "MODEL_OUTPUT_DIR": str(tmp_path),
        "MODEL_MANIFEST_PATH": str(registry_path),
        "DATA_MODE": "local",
        "DATASET_FILE_PATH": "unused.csv",
        "DATASET_ENCODING": "utf-8",
        "DATASET_SEPARATOR": ",",
        "DISTRIBUTED_MODE_ENABLED": False,
    }
    result = RiskService(config).predict_risk(
        {
            "age": 70,
            "gender": 1,
            "bmi": 31,
            "cholesterol": 3,
            "diabetes": 1,
            "hypertension": 1,
            "smoker": 2,
            "alcohol": 1,
            "exercise": 0,
        }
    )

    assert "heart_predicted_probability" in result
    assert "stroke_predicted_probability" in result
    assert result["risk_level"]["code"] in {1, 2, 3, 4, 5}
