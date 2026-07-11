from pathlib import Path

import pandas as pd

from src.services.population_service import PopulationService


def test_population_dashboard_and_follow_up_use_cached_dataframe(tmp_path):
    dataset = tmp_path / "population.csv"
    pd.DataFrame(
        [
            [70, "锦江区", "60-74", 31.2, 1, 1, 3, 2, 0, 1, 1],
            [45, "青羊区", "40-59", 23.5, 0, 0, 1, 0, 1, 0, 0],
            [66, "锦江区", "60-74", 27.1, 1, 0, 2, 1, 0, 0, 1],
        ],
        columns=PopulationService.REQUIRED_COLUMNS,
    ).to_csv(dataset, index=False)

    service = PopulationService({"POPULATION_DATASET_PATH": str(dataset)})
    first = service.dashboard()
    second = PopulationService({"POPULATION_DATASET_PATH": str(dataset)}).dashboard()

    assert first["total_residents"] == 3
    assert first["high_risk_follow_up_count"] == 2
    assert first["districts"][0]["residents"] == 2
    assert len(service.follow_up_list(limit=10)) == 2
    assert first == second
    assert len(PopulationService._dataframe_cache) >= 1


def test_population_service_reports_missing_file(tmp_path):
    service = PopulationService({"POPULATION_DATASET_PATH": str(tmp_path / "missing.csv")})

    try:
        service.dashboard()
    except FileNotFoundError as exc:
        assert "仿真数据" in str(exc)
    else:
        raise AssertionError("Expected a missing population dataset error")
