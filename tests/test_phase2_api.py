import json
import time
from io import BytesIO

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from config import BaseConfig
from src import create_app
from src.ml.calibrated_model import CalibratedRiskModel


FEATURES = [
    "age", "gender", "bmi", "cholesterol", "diabetes", "hypertension",
    "smoker", "alcohol", "exercise",
]


def _write_small_models(directory):
    features = pd.DataFrame(
        [[45, 0, 22, 1, 0, 0, 0, 0, 1], [72, 1, 32, 3, 1, 1, 2, 1, 0]],
        columns=FEATURES,
    )
    for target in ("heart", "stroke"):
        estimator = RandomForestClassifier(n_estimators=5, random_state=42).fit(
            features, [0, 1]
        )
        calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.9], [0, 1])
        joblib.dump(
            CalibratedRiskModel(estimator, calibrator, FEATURES),
            directory / f"{target}.joblib",
        )


def test_upload_finalize_starts_local_analysis_task(tmp_path):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")
        LOCAL_RESULT_ROOT = str(tmp_path / "results")
        MODEL_MANIFEST_PATH = str(tmp_path / "missing-models.json")

    client = create_app(TestConfig).test_client()
    content = (
        "age,region,district,bmi,hypertension\n"
        "55,成都市,武侯区,26,1\n"
        "66,成都市,武侯区,28,1\n"
        "72,成都市,青羊区,30,1\n"
    ).encode("utf-8")
    split = len(content) // 2

    response = client.post(
        "/api/upload/tasks",
        json={
            "filename": "sample.csv",
            "file_size": len(content),
            "total_chunks": 2,
            "data_period": "2026-07",
        },
    )
    assert response.status_code == 200
    task_id = response.get_json()["data"]["task_id"]
    chunk_response = client.post(
        f"/api/upload/tasks/{task_id}/chunks",
        data={"chunk": (BytesIO(content[split:]), "part.csv"), "chunk_index": "1"},
        content_type="multipart/form-data",
    )
    assert chunk_response.status_code == 200, chunk_response.get_json()
    chunk_response = client.post(
        f"/api/upload/tasks/{task_id}/chunks",
        data={"chunk": (BytesIO(content[:split]), "part.csv"), "chunk_index": "0"},
        content_type="multipart/form-data",
    )
    assert chunk_response.status_code == 200, chunk_response.get_json()
    response = client.post(f"/api/upload/tasks/{task_id}/finalize")
    assert response.status_code == 200
    dataset_id = response.get_json()["data"]["dataset_id"]

    status = None
    for _ in range(20):
        status = client.get(f"/api/task/status/{task_id}").get_json()["data"]
        if status["status"] in {"success", "failed"} and status.get("stage") == "completed":
            break
        time.sleep(0.1)
    assert status["status"] == "success"
    result = client.get(f"/api/datasets/{dataset_id}/result")
    assert result.status_code == 200
    assert result.get_json()["data"]["total_residents"] == 3
    assert result.get_json()["data"]["risk_metric_source"] == "unavailable"
    assert result.get_json()["data"]["heart_risk_rate"] is None
    dashboard = client.get("/api/analysis/dashboard")
    dashboard_data = dashboard.get_json()["data"]
    assert dashboard_data["has_data"] is True
    assert dashboard_data["data_source"] == "sample.csv"
    assert dashboard_data["total_residents"] == 3


def test_dashboard_without_uploaded_dataset_returns_waiting_state(tmp_path):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")
        LOCAL_RESULT_ROOT = str(tmp_path / "results")

    client = create_app(TestConfig).test_client()
    response = client.get("/api/analysis/dashboard")
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["has_data"] is False
    assert data["initialized"] is False
    assert data["data_source"] is None
    assert data["total_residents"] == 0
    assert data["coverage_name"] == "等待上传数据"


def test_dataset_preview_masks_sensitive_columns(tmp_path):
    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")
        LOCAL_RESULT_ROOT = str(tmp_path / "results")
        MODEL_MANIFEST_PATH = str(tmp_path / "missing-models.json")

    client = create_app(TestConfig).test_client()
    content = (
        "age,region,district,resident_id,phone\n"
        "55,成都市,武侯区,510100000001,13812345678\n"
    ).encode("utf-8")
    response = client.post(
        "/api/upload/tasks",
        json={
            "filename": "preview.csv",
            "file_size": len(content),
            "total_chunks": 1,
            "data_period": "2026-Q3",
        },
    )
    task_id = response.get_json()["data"]["task_id"]
    client.post(
        f"/api/upload/tasks/{task_id}/chunks",
        data={"chunk": (BytesIO(content), "part.csv"), "chunk_index": "0"},
        content_type="multipart/form-data",
    )
    dataset_id = client.post(f"/api/upload/tasks/{task_id}/finalize").get_json()["data"]["dataset_id"]

    preview = client.get(f"/api/datasets/{dataset_id}/preview").get_json()["data"]
    row = preview["preview_rows"][0]
    assert row["resident_id"] != "510100000001"
    assert row["phone"] == "138****5678"


def test_local_upload_analysis_scores_dual_models(tmp_path):
    _write_small_models(tmp_path)
    manifest = tmp_path / "active_models.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "updated_at": "test-model-v1",
                "feature_columns": FEATURES,
                "models": {"heart": "heart.joblib", "stroke": "stroke.joblib"},
            }
        ),
        encoding="utf-8",
    )

    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        UPLOAD_ROOT = str(tmp_path / "uploads")
        LOCAL_RESULT_ROOT = str(tmp_path / "results")
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(manifest)

    client = create_app(TestConfig).test_client()
    rows = [
        [45, 0, 22, 1, 0, 0, 0, 0, 1, 0, 0, "成都市", "武侯区"],
        [72, 1, 32, 3, 1, 1, 2, 1, 0, 1, 0, "成都市", "武侯区"],
        [55, 0, 25, 2, 0, 1, 1, 0, 1, 0, 0, "成都市", "武侯区"],
        [60, 1, 27, 2, 0, 1, 0, 0, 1, 0, 1, "成都市", "武侯区"],
        [68, 1, 29, 3, 1, 1, 2, 1, 0, 1, 0, "成都市", "武侯区"],
        [50, 0, 23, 1, 0, 0, 0, 0, 1, 0, 0, "成都市", "青羊区"],
    ]
    columns = FEATURES + ["label_heart", "label_stroke", "region", "district"]
    content = pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")
    response = client.post(
        "/api/upload/tasks",
        json={
            "filename": "scored.csv",
            "file_size": len(content),
            "total_chunks": 1,
            "data_period": "2026-07",
        },
    )
    assert response.status_code == 200
    task_id = response.get_json()["data"]["task_id"]
    client.post(
        f"/api/upload/tasks/{task_id}/chunks",
        data={"chunk": (BytesIO(content), "part.csv"), "chunk_index": "0"},
        content_type="multipart/form-data",
    )
    finalized = client.post(f"/api/upload/tasks/{task_id}/finalize")
    assert finalized.status_code == 200
    dataset_id = finalized.get_json()["data"]["dataset_id"]

    status = None
    for _ in range(60):
        status = client.get(f"/api/task/status/{task_id}").get_json()["data"]
        if status["status"] in {"success", "failed"}:
            break
        time.sleep(0.1)
    assert status["status"] == "success", status
    result = client.get(f"/api/datasets/{dataset_id}/result")
    data = result.get_json()["data"]
    assert data["risk_metric_source"] == "model_probability"
    assert data["model_version"] == "test-model-v1"
    assert data["total_residents"] == 6


def test_what_if_requires_complete_matching_feature_payload(tmp_path):
    _write_small_models(tmp_path)
    manifest = tmp_path / "active_models.json"
    manifest.write_text(
        json.dumps({"version": 2, "feature_columns": FEATURES, "models": {"heart": "heart.joblib", "stroke": "stroke.joblib"}}),
        encoding="utf-8",
    )

    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(manifest)

    client = create_app(TestConfig).test_client()
    baseline = {field: 0 for field in FEATURES}
    baseline.update({"age": 55, "bmi": 25, "cholesterol": 1})
    incomplete = dict(baseline)
    incomplete.pop("age")
    response = client.post(
        "/api/risk/what-if",
        json={"baseline": incomplete, "scenario": baseline},
    )
    assert response.status_code == 400


def test_what_if_returns_baseline_scenario_and_delta(tmp_path):
    _write_small_models(tmp_path)
    manifest = tmp_path / "active_models.json"
    manifest.write_text(
        json.dumps({"version": 2, "feature_columns": FEATURES, "models": {"heart": "heart.joblib", "stroke": "stroke.joblib"}}),
        encoding="utf-8",
    )

    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(manifest)

    client = create_app(TestConfig).test_client()
    baseline = dict(zip(FEATURES, [55, 0, 25, 1, 0, 0, 0, 0, 1]))
    scenario = dict(baseline)
    scenario["exercise"] = 0
    response = client.post(
        "/api/risk/what-if",
        json={"baseline": baseline, "scenario": scenario},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["changed_fields"] == ["exercise"]
    assert "baseline" in data and "scenario" in data and "delta" in data


def test_batch_endpoint_returns_masked_record_ids_and_two_rates(tmp_path):
    _write_small_models(tmp_path)
    manifest = tmp_path / "active_models.json"
    manifest.write_text(
        json.dumps({"version": 2, "feature_columns": FEATURES, "models": {"heart": "heart.joblib", "stroke": "stroke.joblib"}}),
        encoding="utf-8",
    )

    class TestConfig(BaseConfig):
        DATABASE_PATH = str(tmp_path / "app.db")
        MODEL_OUTPUT_DIR = str(tmp_path)
        MODEL_MANIFEST_PATH = str(manifest)

    client = create_app(TestConfig).test_client()
    row = dict(zip(FEATURES, [55, 0, 25, 1, 0, 0, 0, 0, 1]))
    row["resident_id"] = "510100000001"
    response = client.post(
        "/api/risk/batch",
        json={"rows": [row], "include_records": True},
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["rows"] == 1
    assert "heart_high_risk_rate" in data
    assert "stroke_high_risk_rate" in data
    assert data["records"][0]["record_id"] != "510100000001"
