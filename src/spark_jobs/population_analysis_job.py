"""Build district and age-group population aggregates in Spark."""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as functions


def run(input_path: str, output_path: str) -> None:
    spark = SparkSession.builder.appName("CardioPopulationAnalysis").getOrCreate()
    dataframe = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
    with_comorbidity = dataframe.withColumn(
        "comorbidity", (functions.col("label_heart") * functions.col("label_stroke"))
    )
    district = with_comorbidity.groupBy("district").agg(
        functions.count("*").alias("residents"),
        functions.avg("label_heart").alias("heart_risk_rate"),
        functions.avg("label_stroke").alias("stroke_risk_rate"),
        functions.avg("comorbidity").alias("comorbidity_rate"),
    )
    district.write.mode("overwrite").option("header", True).csv(f"{output_path}/district")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: spark-submit population_analysis_job.py <input> <output>")
    run(sys.argv[1], sys.argv[2])
