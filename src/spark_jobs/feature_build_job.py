"""Spark DWS feature job with median imputation and a sample-weight field."""

from __future__ import annotations

import sys

from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
from pyspark.sql import functions as functions


NUMERIC_COLUMNS = ["age", "bmi", "cholesterol", "diabetes", "hypertension", "smoker", "alcohol", "exercise", "gender"]


def run(input_path: str, output_path: str) -> None:
    spark = SparkSession.builder.appName("CardioFeatureBuild").getOrCreate()
    dataframe = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
    # Spark 侧使用分布式中位数插补，保证数据链路可在伪分布式环境完整跑通。
    imputer = Imputer(strategy="median", inputCols=NUMERIC_COLUMNS, outputCols=NUMERIC_COLUMNS)
    featured = imputer.fit(dataframe).transform(dataframe)
    featured = featured.withColumn("bmi", functions.round("bmi", 2)).withColumn(
        "sample_weight", functions.lit(1.0)
    )
    featured.write.mode("overwrite").option("header", True).csv(output_path)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: spark-submit feature_build_job.py <input_path> <output_path>")
    run(sys.argv[1], sys.argv[2])
