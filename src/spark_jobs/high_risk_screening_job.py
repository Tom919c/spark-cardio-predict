"""Screen residents requiring priority follow-up in Spark."""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as functions


def run(input_path: str, output_path: str) -> None:
    spark = SparkSession.builder.appName("CardioHighRiskScreening").getOrCreate()
    dataframe = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
    screened = dataframe.filter(
        (functions.col("age") >= 65)
        & ((functions.col("label_heart") == 1) | (functions.col("label_stroke") == 1))
    ).withColumn(
        "risk_score",
        # 风险分仅用于随访优先级排序，不替代两个模型的预测概率。
        functions.col("label_heart")
        + functions.col("label_stroke")
        + functions.col("hypertension")
        + functions.col("diabetes")
        + (functions.col("cholesterol") >= 2).cast("int"),
    )
    screened.orderBy(functions.desc("risk_score"), functions.desc("age")).write.mode(
        "overwrite"
    ).option("header", True).csv(output_path)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: spark-submit high_risk_screening_job.py <input> <output>")
    run(sys.argv[1], sys.argv[2])
