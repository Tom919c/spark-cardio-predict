"""数据服务层：处理本地和分布式模式下的数据集加载、预处理与训练数据构建。"""

from pathlib import Path

import pandas as pd

from src.ml.feature_engineering import CardioFeatureEngineering
from src.utils.validator import DatasetValidator


class DataService:
    """处理本地 CSV 和 HDFS 分布式模式下的数据集加载。"""

    def __init__(self, config):
        self.config = config
        self.dataset_path = config.get("DATASET_FILE_PATH", "")
        self.dataset_encoding = config.get("DATASET_ENCODING", "utf-8")
        self.dataset_separator = config.get("DATASET_SEPARATOR", ",")
        self.distributed_mode_enabled = config.get("DISTRIBUTED_MODE_ENABLED", False)
        self.staging_data_path = config.get("STAGING_DATA_PATH", "")
        self.feature_data_path = config.get("FEATURE_DATA_PATH", "")
        self.hdfs_input_path = config.get("HDFS_INPUT_PATH", "")
        self.hdfs_staging_path = config.get("HDFS_STAGING_PATH", "")
        self.hdfs_feature_path = config.get("HDFS_FEATURE_PATH", "")
        self.feature_columns = config.get("CARDIO_FEATURE_COLUMNS", [])
        self.target_column = config.get("CARDIO_TARGET_COLUMN", "target_disease")
        self.heart_label_column = config.get("HEART_LABEL_COLUMN", "label_heart")
        self.stroke_label_column = config.get("STROKE_LABEL_COLUMN", "label_stroke")
        self.test_size = config.get("TRAIN_TEST_SPLIT_RATIO", 0.2)
        self.random_state = config.get("RANDOM_STATE", 42)

    def get_dataset_profile(self):
        dataset_file = Path(self.dataset_path) if self.dataset_path else None
        return {
            "configured": bool(self.dataset_path),
            "dataset_file_path": self.dataset_path,
            "exists": dataset_file.exists() if dataset_file else False,
            "distributed_mode_enabled": self.distributed_mode_enabled,
            "data_mode": "hdfs" if self.distributed_mode_enabled else "local",
            "staging_data_path": self.staging_data_path,
            "feature_data_path": self.feature_data_path,
            "hdfs_input_path": self.hdfs_input_path,
            "hdfs_staging_path": self.hdfs_staging_path,
            "hdfs_feature_path": self.hdfs_feature_path,
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
        # 本地模式始终读取 DWD CSV；HDFS 模式只在获取训练特征时显式进入 Spark 分支。
        return pd.read_csv(
            self.dataset_path,
            encoding=self.dataset_encoding,
            sep=self.dataset_separator,
        )

    def preprocess_dataset(self):
        profile = self.get_dataset_profile()
        if not profile["configured"] or not profile["exists"]:
            return {**profile, "valid": False}

        dataframe = self.load_dataset()
        validator = DatasetValidator(required_columns=self.feature_columns)
        validation_result = validator.validate_columns(dataframe.columns.tolist())
        if not validation_result["valid"]:
            return {**profile, **validation_result}

        engineer = CardioFeatureEngineering(
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        transformed = engineer.transform(dataframe)
        return {**profile, **validation_result, **transformed}

    def get_training_dataframe(self):
        profile = self.get_dataset_profile()
        if self.distributed_mode_enabled:
            return self._load_distributed_feature_dataframe()

        if not profile["configured"] or not profile["exists"]:
            raise FileNotFoundError("Dataset file is not configured or does not exist.")

        dataframe = self.load_dataset()
        validator = DatasetValidator(required_columns=self.feature_columns)
        validation_result = validator.validate_columns(dataframe.columns.tolist())
        if not validation_result["valid"]:
            raise ValueError(
                f"Dataset validation failed. Missing columns: {validation_result['missing_columns']}"
            )

        engineer = CardioFeatureEngineering(
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # 保留缺失值到训练器；插补器只能在训练子集上拟合，避免数据泄漏。
        return engineer.build_training_dataframe(dataframe, impute=False)

    def _load_distributed_feature_dataframe(self):
        """HDFS 模式下从 Spark 特征输出目录读取训练数据。"""
        if self.hdfs_feature_path:
            from pyspark.sql import SparkSession

            spark = self._create_spark_session()
            dataframe = (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .csv(self.hdfs_feature_path)
            )
            pandas_df = dataframe.toPandas()
            spark.stop()
            return pandas_df

        feature_file = Path(self.feature_data_path) if self.feature_data_path else None
        if feature_file and feature_file.exists():
            return pd.read_csv(feature_file, encoding=self.dataset_encoding)

        raise FileNotFoundError(
            "Distributed mode is enabled, but no HDFS feature path or local feature file is available."
        )

    def _create_spark_session(self):
        from pyspark.sql import SparkSession

        return SparkSession.builder.appName("CardioDistributedDataService").getOrCreate()
