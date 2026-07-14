import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from config import BaseConfig
from src.services.knowledge_service import KnowledgeService
from src.services.phase2_analysis_service import Phase2AnalysisService
from src.services.region_service import RegionService
from src.services.snapshot_service import SnapshotService
from src.utils.privacy import mask_dataframe
from src.dao.hdfs_dao import HDFSDao
from scripts.run_phase2_analysis import _build_source_archive


def test_region_service_uses_district_scope_and_suppresses_small_groups():
    dataframe = pd.DataFrame(
        {
            "region": ["成都市"] * 6,
            "district": ["武侯区"] * 5 + ["青羊区"],
            "age": [50, 51, 52, 53, 54, 60],
            "label_heart": [1, 0, 0, 1, 0, 1],
            "label_stroke": [0, 0, 1, 0, 0, 1],
        }
    )
    result = RegionService(min_group_size=5).aggregate(dataframe)

    assert result["coverage"]["coverage_level"] == "district"
    assert result["coverage"]["map_name"] == "chengdu"
    assert result["regions"][-1]["suppressed"] is True


def test_region_service_uses_table_for_single_district():
    dataframe = pd.DataFrame(
        {
            "region": ["成都市"] * 6,
            "district": ["武侯区"] * 6,
            "age": [45, 46, 47, 48, 49, 50],
        }
    )

    coverage = RegionService().infer_coverage(dataframe)

    assert coverage["coverage_level"] == "table"
    assert coverage["coverage_name"] == "武侯区"
    assert coverage["map_available"] is False


def test_region_service_does_not_invent_label_rates_when_labels_are_missing():
    dataframe = pd.DataFrame(
        {
            "region": ["成都市"] * 6,
            "district": ["武侯区"] * 6,
            "age": [45, 46, 47, 48, 49, 50],
            "heart_probability": [0.7, 0.1, 0.2, 0.8, 0.1, 0.1],
            "stroke_probability": [0.1, 0.7, 0.2, 0.1, 0.1, 0.1],
        }
    )
    result = RegionService().aggregate(
        dataframe,
        model_scores={
            "heart_probability": dataframe["heart_probability"],
            "stroke_probability": dataframe["stroke_probability"],
        },
    )

    row = result["regions"][0]
    assert row["heart_rate"] is None
    assert row["stroke_rate"] is None


def test_sensitive_preview_fields_are_masked():
    dataframe = pd.DataFrame(
        {"resident_id": ["510100000001"], "phone": ["13812345678"], "age": [50]}
    )

    masked = mask_dataframe(dataframe)

    assert masked.loc[0, "resident_id"] != dataframe.loc[0, "resident_id"]
    assert masked.loc[0, "phone"] == "138****5678"
    assert masked.loc[0, "age"] == 50


def test_hdfs_dao_supports_webhdfs_status_existence_check():
    class StatusOnlyClient:
        def status(self, path, strict=True):
            return {"pathSuffix": "summary.json"} if path == "/result/summary.json" else None

    dao = HDFSDao(StatusOnlyClient())

    assert dao.exists("/result/summary.json") is True
    assert dao.exists("/result/missing.json") is False


def test_hdfs_dao_uses_insecure_client_makedirs():
    class MakdirsClient:
        def __init__(self):
            self.paths = []

        def makedirs(self, path):
            self.paths.append(path)

    client = MakdirsClient()
    HDFSDao(client).ensure_dir("/upload/task")

    assert client.paths == ["/upload/task"]


def test_snapshot_service_requires_compatible_model_scope():
    snapshots = [
        {"data_period": "2026-01", "coverage_level": "district", "coverage_name": "成都", "model_version": "v1", "heart_risk_rate": 10},
        {"data_period": "2026-02", "coverage_level": "district", "coverage_name": "成都", "model_version": "v1", "heart_risk_rate": 12},
    ]
    result = SnapshotService().compare(snapshots)

    assert result["available"] is True
    assert [item["data_period"] for item in result["series"]] == ["2026-01", "2026-02"]


def test_knowledge_service_returns_versioned_rules(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "rules": [
                    {"rule_id": "smoke", "factor": "smoker", "risk_levels": [1], "advice": "戒烟", "source": "test"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = KnowledgeService(path).build_plan({"smoker": 2}, {"code": 1})

    assert result["knowledge_version"] == "test-v1"
    assert result["suggestions"][0]["rule_id"] == "smoke"


def test_curated_knowledge_base_preserves_sources_and_safety_boundary():
    path = Path("resources/knowledge/intervention_rules.json")
    content = json.loads(path.read_text(encoding="utf-8"))

    assert len(content["rules"]) == 168
    assert len({rule["rule_id"] for rule in content["rules"]}) == 168
    assert {item["namespace"] for item in content["sources"]} == {
        "stroke-report-2024",
        "stroke-primary-2021",
        "cvd-primary-2020",
        "diet-2022",
    }
    assert all(rule.get("source_doc") for rule in content["rules"])
    assert all(rule.get("source_section") for rule in content["rules"])
    assert any(rule["review_mode"] == "manual" for rule in content["rules"])
    assert all(
        rule["automation_safe"] is True
        for rule in content["rules"]
        if rule["review_mode"] == "auto"
    )


def test_knowledge_service_only_returns_automatic_rules_and_supports_alcohol():
    service = KnowledgeService(Path("resources/knowledge/intervention_rules.json"))
    result = service.build_plan(
        {"alcohol": 1, "hypertension": 0, "cholesterol": 1, "exercise": 1},
        {"code": 2},
    )

    assert result["suggestions"]
    assert all(item["automation_safe"] is True for item in result["suggestions"])
    assert any(item["rule_id"] == "diet-2022-diet-rule-014" for item in result["suggestions"])
    assert not any("阿司匹林" in item["text"] for item in result["suggestions"])


def test_high_risk_follow_up_is_not_dropped_when_many_factors_match():
    service = KnowledgeService(Path("resources/knowledge/intervention_rules.json"))
    result = service.build_plan(
        {
            "alcohol": 1,
            "hypertension": 1,
            "cholesterol": 3,
            "diabetes": 1,
            "smoker": 1,
            "exercise": 0,
            "bmi": 30,
        },
        {"code": 5},
    )

    assert result["suggestions"][0]["rule_id"] == "high-risk-follow-up"


def test_knowledge_service_builds_three_friendly_deduplicated_actions():
    service = KnowledgeService(Path("resources/knowledge/intervention_rules.json"))
    result = service.build_plan(
        {
            "hypertension": 1,
            "cholesterol": 3,
            "diabetes": 1,
            "smoker": 2,
            "alcohol": 1,
            "exercise": 0,
            "bmi": 30,
        },
        {"code": 5},
    )

    assert len(result["actions"]) == 3
    assert len({item["domain"] for item in result["actions"]}) == 3
    assert result["focus_summary"].startswith("您现在最需要关注的是")
    assert "2 周内" in result["follow_up"]
    assert "120" in result["emergency_warning"]


def test_high_bmi_does_not_trigger_exercise_advice_when_activity_is_regular():
    service = KnowledgeService(Path("resources/knowledge/intervention_rules.json"))
    result = service.build_plan(
        {"bmi": 28, "exercise": 1, "hypertension": 0, "cholesterol": 1},
        {"code": 2},
    )

    assert [item["domain"] for item in result["actions"]] == ["bmi"]
    assert "体重" in result["actions"][0]["text"]


def test_phase2_analysis_service_writes_summary(tmp_path):
    dataframe = pd.DataFrame(
        {
            "age": [50, 60, 70, 80, 55, 65],
            "region": ["成都市"] * 6,
            "district": ["武侯区"] * 5 + ["青羊区"],
            "bmi": [24, 26, 30, 28, 23, 27],
            "hypertension": [0, 1, 1, 1, 0, 1],
            "diabetes": [0, 0, 1, 1, 0, 0],
            "cholesterol": [1, 2, 3, 2, 1, 2],
            "smoker": [0, 2, 1, 2, 0, 1],
            "exercise": [1, 0, 0, 0, 1, 0],
            "label_heart": [0, 1, 1, 1, 0, 1],
            "label_stroke": [0, 0, 1, 1, 0, 0],
        }
    )
    source = tmp_path / "sample.csv"
    dataframe.to_csv(source, index=False)

    class TestConfig(BaseConfig):
        MODEL_MANIFEST_PATH = str(tmp_path / "missing.json")
        LOCAL_ANALYSIS_CHUNK_SIZE = 2

    result = Phase2AnalysisService(TestConfig.as_dict()).analyze_file(
        source, tmp_path / "result", {"dataset_id": "ds-test", "data_period": "2026-07"}
    )

    assert result["total_residents"] == 6
    assert result["coverage_level"] == "district"
    assert (tmp_path / "result" / "summary.json").exists()


def test_local_region_metrics_prefer_labels_over_predictions():
    service = Phase2AnalysisService(BaseConfig.as_dict())
    rows = service._build_group_rows(
        {
            "武侯区": {
                "residents": 5,
                "heart_positive": 0,
                "stroke_positive": 0,
                "comorbidity": 0,
                "heart_high": 5,
                "stroke_high": 5,
                "predicted_comorbidity": 5,
            }
        },
        {
            "coverage_level": "district",
            "coverage_name": "成都市",
        },
        metric_available=True,
        labels_available=True,
    )

    assert rows[0]["heart_rate"] == 0.0
    assert rows[0]["stroke_rate"] == 0.0
    assert rows[0]["comorbidity_rate"] == 0.0


def test_spark_source_archive_contains_custom_model_module():
    with tempfile.TemporaryDirectory() as directory:
        archive = _build_source_archive(Path(directory))
        with zipfile.ZipFile(archive) as package:
            assert "src/ml/calibrated_model.py" in package.namelist()
