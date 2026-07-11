import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from src import create_app
from src.ml.calibrated_model import CalibratedRiskModel
from config import BaseConfig


FEATURES = [
    "age", "gender", "bmi", "cholesterol", "diabetes", "hypertension",
    "smoker", "alcohol", "exercise",
]


def _write_test_models(directory):
    features = pd.DataFrame(
        [[45, 0, 22, 1, 0, 0, 0, 0, 1], [72, 1, 32, 3, 1, 1, 2, 1, 0]],
        columns=FEATURES,
    )
    for target in ("heart", "stroke"):
        estimator = RandomForestClassifier(n_estimators=5, random_state=42).fit(features, [0, 1])
        calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.9], [0, 1])
        joblib.dump(
            CalibratedRiskModel(estimator, calibrator, FEATURES),
            directory / f"{target}.joblib",
        )


def test_health_and_page_routes_are_available():
    app = create_app()
    client = app.test_client()

    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/dashboard").status_code == 200
    assert client.get("/risk-report").status_code == 200


def test_risk_predict_rejects_incomplete_payload():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/risk/predict", json={"age": 55})

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_risk_predict_reports_unavailable_models():
    app = create_app()
    response = app.test_client().post(
        "/api/risk/predict",
        json={"age": 70, "gender": 1, "bmi": 31, "cholesterol": 3,
              "diabetes": 1, "hypertension": 1, "smoker": 2,
              "alcohol": 1, "exercise": 0},
    )

    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert "模型" in response.get_json()["message"]


def test_risk_predict_returns_two_probabilities(tmp_path):
    _write_test_models(tmp_path)
    manifest = tmp_path / "active_models.json"
    manifest.write_text(
        '{"version":2,"feature_columns":' + str(FEATURES).replace("'", '"') +
        ',"models":{"heart":"heart.joblib","stroke":"stroke.joblib"}}',
        encoding="utf-8",
    )

    class TestConfig(BaseConfig):
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(manifest)

    app = create_app(TestConfig)
    response = app.test_client().post(
        "/api/risk/predict",
        json={"age": 70, "gender": 1, "bmi": 31, "cholesterol": 3,
              "diabetes": 1, "hypertension": 1, "smoker": 2,
              "alcohol": 1, "exercise": 0},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert "heart_predicted_probability" in data
    assert "stroke_predicted_probability" in data
    assert "combined_shap_summary" in data
    assert data["heart_predicted_probability"] >= 0
    assert data["stroke_predicted_probability"] >= 0
