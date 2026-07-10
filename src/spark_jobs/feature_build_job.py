import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round


class CardioFeatureBuildJob:
    """Builds distributed training features for the cardiovascular dataset."""

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        spark = self._create_spark_session()
        dataframe = self._read_input(spark)
        feature_dataframe = self._build_features(dataframe)
        self._write_output(feature_dataframe)
        spark.stop()

    def _create_spark_session(self):
        return SparkSession.builder.appName("CardioFeatureBuildJob").getOrCreate()

    def _read_input(self, spark):
        return (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(self.input_path)
        )

    def _build_features(self, dataframe):
        feature_df = dataframe.withColumn(
            "age_years", spark_round(col("age") / 365, 1)
        )
        feature_df = feature_df.withColumn(
            "bmi",
            spark_round(
                col("weight") / ((col("height") / 100) * (col("height") / 100)), 2
            ),
        )
        return feature_df.select(
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "age_years",
            "bmi",
            "cardio",
        )

    def _write_output(self, dataframe):
        (
            dataframe.write.mode("overwrite")
            .option("header", True)
            .csv(self.output_path)
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: spark-submit feature_build_job.py <input_path> <output_path>"
        )

    job = CardioFeatureBuildJob(input_path=sys.argv[1], output_path=sys.argv[2])
    job.run()
