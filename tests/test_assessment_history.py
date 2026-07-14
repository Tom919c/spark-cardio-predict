import json

from config import BaseConfig
from src import create_app
from src.dao.database_dao import DatabaseDAO
from src.services.assessment_record_service import AssessmentRecordService


SAMPLE = {
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

RESULT = {
    "heart_predicted_probability": 0.42,
    "stroke_predicted_probability": 0.21,
    "risk_level": {"code": 3, "name": "III级：中危", "color": "#ea580c"},
    "final_category": 1,
    "intervention_knowledge_version": "rules-1",
    "combined_shap_summary": {"available": False},
}


def test_assessment_service_hashes_client_and_serializes_json(tmp_path):
    database_path = str(tmp_path / "app.db")
    config = {
        "DATABASE_TYPE": "sqlite",
        "DATABASE_PATH": database_path,
        "SECRET_KEY": "test-secret",
        "ASSESSMENT_HASH_KEY": "assessment-secret",
        "MODEL_MANIFEST_PATH": str(tmp_path / "missing.json"),
    }
    dao = DatabaseDAO(database_path)
    service = AssessmentRecordService(config, database_dao=dao)

    created = service.create("browser-client-001", SAMPLE, RESULT)
    stored = dao.get_assessment(created["assessment_id"])

    assert stored["client_hash"] != "browser-client-001"
    assert len(stored["client_hash"]) == 64
    assert json.loads(stored["input_json"])["age"] == 70
    assert json.loads(stored["result_json"])["final_category"] == 1
    assert "client_hash" not in created
    assert "input_json" not in created
    assert service.list_for_client("browser-client-001")[0]["assessment_id"] == created["assessment_id"]


def test_predict_persists_history_and_enforces_client_ownership(tmp_path, monkeypatch):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        ASSESSMENT_HASH_KEY = "assessment-test-key"
        MODEL_MANIFEST_PATH = str(tmp_path / "missing.json")

    monkeypatch.setattr(
        "src.controllers.risk_controller.RiskService.predict_risk",
        lambda self, sample: dict(RESULT),
    )
    client = create_app(TestConfig).test_client()
    headers = {"X-Client-ID": "browser-client-001"}

    prediction = client.post("/api/risk/predict", json=SAMPLE, headers=headers)
    assert prediction.status_code == 200
    assessment_id = prediction.get_json()["data"]["assessment_id"]

    history = client.get("/api/assessments", headers=headers)
    assert history.status_code == 200
    items = history.get_json()["data"]["items"]
    assert items[0]["assessment_id"] == assessment_id
    assert items[0]["risk_level"]["code"] == 3
    assert "client_hash" not in items[0]

    detail = client.get(f"/api/assessments/{assessment_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.get_json()["data"]["heart_probability"] == 0.42

    denied = client.get(
        f"/api/assessments/{assessment_id}",
        headers={"X-Client-ID": "different-client-002"},
    )
    assert denied.status_code == 404


def test_predict_generates_retrievable_client_id_when_header_is_missing(tmp_path, monkeypatch):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        ASSESSMENT_HASH_KEY = "assessment-test-key"
        MODEL_MANIFEST_PATH = str(tmp_path / "missing.json")

    monkeypatch.setattr(
        "src.controllers.risk_controller.RiskService.predict_risk",
        lambda self, sample: dict(RESULT),
    )
    client = create_app(TestConfig).test_client()
    response = client.post("/api/risk/predict", json=SAMPLE)

    assert response.status_code == 200
    data = response.get_json()["data"]
    generated_client_id = data["assessment_client_id"]
    history = client.get(
        "/api/assessments", headers={"X-Client-ID": generated_client_id}
    )
    assert history.status_code == 200
    assert history.get_json()["data"]["items"][0]["assessment_id"] == data["assessment_id"]


def test_history_requires_valid_client_id(tmp_path):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        ASSESSMENT_HASH_KEY = "assessment-test-key"

    client = create_app(TestConfig).test_client()
    assert client.get("/api/assessments").status_code == 400
    assert client.get(
        "/api/assessments", headers={"X-Client-ID": "short"}
    ).status_code == 400
