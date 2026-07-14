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
    assert client.get("/api/risk/train").status_code == 405
    dashboard = client.get("/dashboard").get_data(as_text=True)
    assert 'id="phase2DataPeriod"' in dashboard
    assert 'id="phase2DatabaseEngine"' in dashboard


def test_personal_report_flow_has_one_primary_explanation_path():
    client = create_app().test_client()

    risk_page = client.get("/risk-report").get_data(as_text=True)
    result_page = client.get("/result-report").get_data(as_text=True)
    shap_page = client.get("/shap-analysis").get_data(as_text=True)

    assert "查看完整风险报告" in risk_page
    assert "查看完整风险原因分析" not in risk_page
    assert "maximum-scale" not in risk_page
    assert '<main class="page-wrap" id="main-content">' in risk_page
    assert "专业模型解释" in result_page
    assert "返回完整报告" in shap_page
    assert "不代表医学因果关系" in shap_page


def test_capabilities_report_runtime_topology(tmp_path):
    class TestConfig(BaseConfig):
        DATA_MODE = "local"
        DATABASE_TYPE = "sqlite"
        DATABASE_PATH = str(tmp_path / "app.db")
        MODEL_OUTPUT_DIR = str(tmp_path / "models")
        MODEL_MANIFEST_PATH = str(tmp_path / "models" / "missing.json")

    response = create_app(TestConfig).test_client().get("/api/capabilities")

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["database"]["type"] == "sqlite"
    assert data["database"]["supported_types"] == ["sqlite", "mysql"]
    assert data["data_mode"] == "local"
    assert data["storage_engine"] == "local_filesystem"
    assert data["compute_engine"] == "local_pandas"
    assert data["hdfs"]["enabled"] is False
    assert "cli_detected" in data["hdfs"]
    assert data["spark"]["analysis_script_exists"] is True
    assert data["models"]["ready"] is False


def test_risk_predict_rejects_incomplete_payload():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/risk/predict", json={"age": 55})

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_risk_predict_reports_unavailable_models(tmp_path):
    class TestConfig(BaseConfig):
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(tmp_path / "active_models.json")

    app = create_app(TestConfig)
    response = app.test_client().post(
        "/api/risk/predict",
        json={"age": 70, "gender": 1, "bmi": 31, "cholesterol": 3,
              "diabetes": 1, "hypertension": 1, "smoker": 2,
              "alcohol": 1, "exercise": 0},
    )

    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert "模型" in response.get_json()["message"]


def test_risk_predict_rejects_invalid_values():
    app = create_app()
    response = app.test_client().post(
        "/api/risk/predict",
        json={"age": 120, "gender": 1, "bmi": 31, "cholesterol": 3,
              "diabetes": 1, "hypertension": 1, "smoker": 2,
              "alcohol": 1, "exercise": 0},
    )

    assert response.status_code == 400
    assert "字段取值无效" in response.get_json()["message"]

    auxiliary_response = app.test_client().post(
        "/api/risk/predict",
        json={"age": 70, "gender": 1, "bmi": 31, "cholesterol": 3,
              "diabetes": 1, "hypertension": 1, "smoker": 2,
              "alcohol": 1, "exercise": 0, "systolic_bp": 999},
    )
    assert auxiliary_response.status_code == 400
    assert "systolic_bp" in auxiliary_response.get_json()["message"]


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
    assert data["intervention_plan"]["risk_level"]["code"] in {1, 2, 3, 4, 5}
    assert data["model_scope"]["type"] == "screening_proxy"
    assert data["intervention_plan"]["actions"]


def test_train_estimate_reports_missing_dataset_clearly(tmp_path):
    class TestConfig(BaseConfig):
        DATASET_FILE_PATH = str(tmp_path / "missing.csv")

    response = create_app(TestConfig).test_client().get("/api/risk/train-estimate")

    assert response.status_code == 400
    assert "Dataset" in response.get_json()["message"]
