#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# WSL2/VMware 只在此处加载项目 .env，避免每个人在脚本内重复修改路径。
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

SPARK_SUBMIT_BIN="${SPARK_SUBMIT_BIN:-spark-submit}"
HDFS_RAW_PATH="${HDFS_RAW_PATH:-/user/${USER}/cardio/raw}"
HDFS_STAGING_PATH="${HDFS_STAGING_PATH:-/user/${USER}/cardio/staging}"
HDFS_FEATURE_PATH="${HDFS_FEATURE_PATH:-/user/${USER}/cardio/feature}"
INPUT_FILE="${1:-${HDFS_INPUT_PATH:-$HDFS_RAW_PATH/chengdu_resident_health_simulated.csv}}"

cd "$PROJECT_DIR"
"$SPARK_SUBMIT_BIN" src/spark_jobs/etl_job.py "$INPUT_FILE" "$HDFS_STAGING_PATH"
"$SPARK_SUBMIT_BIN" src/spark_jobs/feature_build_job.py "$HDFS_STAGING_PATH" "$HDFS_FEATURE_PATH"
"$SPARK_SUBMIT_BIN" src/spark_jobs/population_analysis_job.py "$HDFS_FEATURE_PATH" "$HDFS_FEATURE_PATH/ads"
"$SPARK_SUBMIT_BIN" src/spark_jobs/high_risk_screening_job.py "$HDFS_FEATURE_PATH" "$HDFS_FEATURE_PATH/follow_ups"
