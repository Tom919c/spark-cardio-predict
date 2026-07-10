from pathlib import Path

import pandas as pd

from src.ml.feature_engineering import CardioFeatureEngineering
from src.utils.validator import DatasetValidator


class DataService:
    """Handles local dataset loading and preprocessing for the first project stage."""

    def __init__(self, config):
        self.config = config
        self.dataset_path = config.get("DATASET_FILE_PATH", "")
        self.dataset_encoding = config.get("DATASET_ENCODING", "utf-8")
        self.dataset_separator = config.get("DATASET_SEPARATOR", ";")
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])
        self.target_column = config.get("CARDIO_TARGET_COLUMN", "cardio")
        self.test_size = config.get("TRAIN_TEST_SPLIT_RATIO", 0.2)
        self.random_state = config.get("RANDOM_STATE", 42)

    def get_dataset_profile(self):
        dataset_file = Path(self.dataset_path) if self.dataset_path else None
        return {
            "configured": bool(self.dataset_path),
            "dataset_file_path": self.dataset_path,
            "exists": dataset_file.exists() if dataset_file else False,
            "encoding": self.dataset_encoding,
            "separator": self.dataset_separator,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }

    def preview_dataset(self, rows=5):
        profile = self.get_dataset_profile()
        if not profile["configured"] or not profile["exists"]:
            return profile

        dataframe = self.load_dataset()
        preview_frame = dataframe.head(rows)
        return {
            **profile,
            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "columns": dataframe.columns.tolist(),
            "preview_rows": preview_frame.to_dict(orient="records"),
        }

    def load_dataset(self):
        return pd.read_csv(
            self.dataset_path,
            encoding=self.dataset_encoding,
            sep=self.dataset_separator,
        )

    def preprocess_dataset(self):
        profile = self.get_dataset_profile()
        if not profile["configured"] or not profile["exists"]:
            return {
                **profile,
                "valid": False,
            }

        dataframe = self.load_dataset()
        validator = DatasetValidator(
            required_columns=self.feature_columns + [self.target_column]
        )
        validation_result = validator.validate_columns(dataframe.columns.tolist())
        if not validation_result["valid"]:
            return {
                **profile,
                **validation_result,
            }

        engineer = CardioFeatureEngineering(
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        transformed = engineer.transform(dataframe)
        return {
            **profile,
            **validation_result,
            **transformed,
        }
