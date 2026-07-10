#!/bin/bash

# Example distributed processing flow:
# 1. ETL clean raw cardio data
# 2. Build distributed feature dataset

SPARK_SUBMIT_BIN="${SPARK_SUBMIT_BIN:-spark-submit}"
RAW_INPUT_PATH="${RAW_INPUT_PATH:-data/raw/datasets/cardio_train.csv}"
STAGING_OUTPUT_PATH="${STAGING_OUTPUT_PATH:-data/staging/cardio_staging}"
FEATURE_OUTPUT_PATH="${FEATURE_OUTPUT_PATH:-data/feature/cardio_features}"

$SPARK_SUBMIT_BIN src/spark_jobs/etl_job.py "$RAW_INPUT_PATH" "$STAGING_OUTPUT_PATH"
$SPARK_SUBMIT_BIN src/spark_jobs/feature_build_job.py "$STAGING_OUTPUT_PATH" "$FEATURE_OUTPUT_PATH"
