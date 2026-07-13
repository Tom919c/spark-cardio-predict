"""Spark ETL job for the standard nine features and two independent labels."""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as functions


FEATURE_COLUMNS = [
    "age", "gender", "bmi", "cholesterol", "diabetes", "hypertension",
    "smoker", "alcohol", "exercise",
]

# 这些字段不参与模型训练，但供 B 端群体分析和重点随访继续使用。
ANALYSIS_COLUMNS = [
    "resident_id", "region", "district", "community_id", "age_group",
]


def run(input_path: str, output_path: str, separator: str = ",") -> None:
    spark = SparkSession.builder.appName("CardioStandardETL").getOrCreate()
    source = spark.read.option("header", True).option("sep", separator).option(
        "inferSchema", True
    ).csv(input_path)
    dataframe = _ensure_labels(source)
    cleaned = _clean_values(dataframe)
    cleaned.write.mode("overwrite").option("header", True).csv(output_path)
    spark.stop()


def _ensure_labels(dataframe):
    columns = set(dataframe.columns)
    result = dataframe
    if "label_heart" not in columns:
        if "target_disease" not in columns:
            raise ValueError("Input must contain label_heart or target_disease.")
        result = result.withColumn(
            "label_heart", (functions.col("target_disease") == 1).cast("int")
        )
    if "label_stroke" not in columns:
        if "target_disease" not in columns:
            raise ValueError("Input must contain label_stroke or target_disease.")
        result = result.withColumn(
            "label_stroke", (functions.col("target_disease") == 2).cast("int")
        )
    return result


def _clean_values(dataframe):
    cleaned = dataframe
    for column in ("age", "bmi", "cholesterol"):
        cleaned = cleaned.withColumn(
            column,
            functions.when(functions.col(column) > 0, functions.col(column)),
        )
    for column in ("gender", "diabetes", "hypertension", "alcohol", "exercise"):
        cleaned = cleaned.withColumn(
            column,
            functions.when(functions.col(column).isin(0, 1), functions.col(column)),
        )
    cleaned = cleaned.withColumn(
        "smoker", functions.when(functions.col("smoker").isin(0, 1, 2), functions.col("smoker"))
    ).withColumn(
        "cholesterol",
        functions.when(functions.col("cholesterol").isin(1, 2, 3), functions.col("cholesterol")),
    )
    # ETL 只标准化非法值为 null；缺失值留给 DWS 统一插补，避免直接删行损失样本。
    # 分析字段按输入实际存在情况透传，兼容 DWD 训练表和 DWS 仿真人口表。
    available_analysis_columns = [
        column for column in ANALYSIS_COLUMNS if column in cleaned.columns
    ]
    return cleaned.select(
        *FEATURE_COLUMNS,
        "label_heart",
        "label_stroke",
        *available_analysis_columns,
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise ValueError("Usage: spark-submit etl_job.py <input_path> <output_path> [separator]")
    run(*sys.argv[1:])
