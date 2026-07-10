import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class CardioETLJob:
    """Runs distributed ETL cleaning for the cardiovascular dataset."""

    def __init__(self, input_path, output_path, separator=";"):
        self.input_path = input_path
        self.output_path = output_path
        self.separator = separator

    def run(self):
        spark = self._create_spark_session()
        dataframe = self._read_input(spark)
        cleaned_dataframe = self._clean_dataframe(dataframe)
        self._write_output(cleaned_dataframe)
        spark.stop()

    def _create_spark_session(self):
        return SparkSession.builder.appName("CardioETLJob").getOrCreate()

    def _read_input(self, spark):
        return (
            spark.read.option("header", True)
            .option("sep", self.separator)
            .option("inferSchema", True)
            .csv(self.input_path)
        )

    def _clean_dataframe(self, dataframe):
        cleaned = dataframe.dropDuplicates()
        for field_name in ["age", "height", "weight", "ap_hi", "ap_lo"]:
            cleaned = cleaned.filter(col(field_name) > 0)
        cleaned = cleaned.filter(col("ap_hi") >= col("ap_lo"))
        return cleaned

    def _write_output(self, dataframe):
        (
            dataframe.write.mode("overwrite")
            .option("header", True)
            .csv(self.output_path)
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: spark-submit etl_job.py <input_path> <output_path>")

    job = CardioETLJob(input_path=sys.argv[1], output_path=sys.argv[2])
    job.run()
