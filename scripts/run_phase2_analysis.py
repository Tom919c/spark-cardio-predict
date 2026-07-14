"""阶段二 HDFS/Spark 分析入口。

Spark 负责读取 HDFS 原始数据、双模型批量评分和小结果聚合，最终只把
ADS 汇总结果写回 HDFS。输出契约与本地 ``Phase2AnalysisService`` 对齐，
供当前运行环境中的 Flask 服务读取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path


FEATURE_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "cholesterol",
    "diabetes",
    "hypertension",
    "smoker",
    "alcohol",
    "exercise",
]
HIGH_RISK_THRESHOLD = 0.60
DEFAULT_MIN_GROUP_SIZE = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model-manifest", default=os.getenv("MODEL_MANIFEST_PATH", ""))
    parser.add_argument(
        "--score-engine",
        choices=("pandas",),
        default=os.getenv("SPARK_SCORE_ENGINE", "pandas"),
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=int(os.getenv("PRIVACY_MIN_GROUP_SIZE", DEFAULT_MIN_GROUP_SIZE)),
    )
    return parser.parse_args()


def main():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise SystemExit("HDFS 模式需要安装 PySpark。") from exc

    args = parse_args()
    if args.min_group_size < 1:
        raise SystemExit("--min-group-size 必须为正整数。")

    with tempfile.TemporaryDirectory(prefix="cardiospark-spark-") as runtime_dir:
        source_archive = _build_source_archive(Path(runtime_dir))
        spark = SparkSession.builder.appName(f"CardioSparkPhase2-{args.task_id}").getOrCreate()
        spark.sparkContext.addPyFile(str(source_archive))
        try:
            frame = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
            total = frame.count()
            if total == 0:
                raise ValueError("CSV 文件为空。")
            columns = set(frame.columns)
            field_mapping = _resolve_field_mapping(frame.columns)
            for canonical, source in field_mapping.items():
                if source != canonical and canonical not in columns:
                    frame = frame.withColumnRenamed(source, canonical)
            columns = set(frame.columns)
            if "age" not in columns:
                raise ValueError("上传数据至少需要 age 字段。")
            if not any(
                field in columns
                for field in ("province", "city", "region", "district", "community", "community_id")
            ):
                raise ValueError("上传数据至少需要一个区域字段。")

            coverage = _infer_coverage(frame, columns)
            scored_frame, prediction_enabled, model_version = _score_with_dual_models(
                frame, args.model_manifest, columns, spark, args.score_engine
            )
            scored_columns = set(scored_frame.columns)
            labels_available = {"label_heart", "label_stroke"}.issubset(columns)
            prediction_enabled = {"heart_probability", "stroke_probability"}.issubset(scored_columns)
            metric_source = (
                "model_probability"
                if prediction_enabled
                else "labels"
                if labels_available
                else "unavailable"
            )
            metric_available = metric_source != "unavailable"
            use_labels = labels_available and not prediction_enabled

            region_field = coverage.get("region_field")
            working = scored_frame
            if not region_field or region_field not in working.columns:
                region_field = "_phase2_region"
                working = working.withColumn(region_field, F.lit("全部数据"))
            working = working.withColumn(
                region_field,
                F.when(
                    F.col(region_field).isNull()
                    | (F.trim(F.col(region_field).cast("string")) == ""),
                    F.lit("未知"),
                ).otherwise(F.col(region_field).cast("string")),
            )

            heart_metric = _metric_expression(working, "heart", use_labels, scored_columns)
            stroke_metric = _metric_expression(working, "stroke", use_labels, scored_columns)
            comorbidity = _comorbidity_expression(working, use_labels, scored_columns)
            follow_up = F.when((heart_metric == 1) | (stroke_metric == 1), 1).otherwise(0)

            grouped = working.withColumn("_heart_metric", heart_metric).withColumn(
                "_stroke_metric", stroke_metric
            ).withColumn("_comorbidity", comorbidity).withColumn("_follow_up", follow_up)
            grouped = grouped.groupBy(region_field).agg(
                F.count(F.lit(1)).alias("residents"),
                F.sum("_heart_metric").alias("heart_numerator"),
                F.sum("_stroke_metric").alias("stroke_numerator"),
                F.sum("_comorbidity").alias("comorbidity_numerator"),
            )
            region_rows = _region_rows(grouped.collect(), coverage, args.min_group_size, metric_available)

            region_output = grouped.select(
                F.col(region_field).alias("region"),
                F.when(F.col("residents") >= args.min_group_size, F.col("residents")).alias("residents"),
                F.when(
                    metric_available & (F.col("residents") >= args.min_group_size),
                    F.col("heart_numerator") / F.col("residents") * 100,
                ).alias("heart_rate"),
                F.when(
                    metric_available & (F.col("residents") >= args.min_group_size),
                    F.col("stroke_numerator") / F.col("residents") * 100,
                ).alias("stroke_rate"),
                F.when(
                    metric_available & (F.col("residents") >= args.min_group_size),
                    F.col("comorbidity_numerator") / F.col("residents") * 100,
                ).alias("comorbidity_rate"),
            )
            region_output.coalesce(1).write.mode("overwrite").option("header", True).csv(
                f"{args.output}/region_summary"
            )

            age_groups = _age_groups(working, use_labels, scored_columns)
            risk_factors = _risk_factors(working)
            follow_ups = _follow_up_records(working, prediction_enabled, scored_columns)
            invalid_rows = _invalid_row_count(working).collect()[0][0] or 0
            # Recompute global numerators from the row-level frame. This avoids collecting
            # resident records and keeps the result independent of the chosen region field.
            global_values = working.withColumn("_heart_metric", heart_metric).withColumn(
                "_stroke_metric", stroke_metric
            ).withColumn("_comorbidity", comorbidity).withColumn("_follow_up", follow_up).agg(
                F.sum("_heart_metric").alias("heart"),
                F.sum("_stroke_metric").alias("stroke"),
                F.sum("_comorbidity").alias("comorbidity"),
                F.sum("_follow_up").alias("follow_up"),
            ).collect()[0].asDict()
            heart_rate = _percent(global_values.get("heart"), total) if metric_available else None
            stroke_rate = _percent(global_values.get("stroke"), total) if metric_available else None
            comorbidity_rate = _percent(global_values.get("comorbidity"), total) if metric_available else None

            summary = {
                "task_id": args.task_id,
                "field_mapping": field_mapping,
                "total_residents": int(total),
                "invalid_rows": int(invalid_rows),
                "labels_available": labels_available,
                "risk_metric_source": metric_source,
                "model_version": model_version if prediction_enabled else "labels-only",
                "feature_version": "phase2-feature-contract-v1",
                "coverage_level": coverage["coverage_level"],
                "coverage_name": coverage["coverage_name"],
                "map_name": coverage["map_name"],
                "map_available": coverage["map_available"],
                "coverage": coverage,
                "districts": region_rows,
                "region_analysis": region_rows,
                "age_groups": age_groups,
                "risk_factors": risk_factors,
                "heart_risk_rate": heart_rate,
                "stroke_risk_rate": stroke_rate,
                "comorbidity_rate": comorbidity_rate,
                "high_risk_follow_up_count": int(global_values.get("follow_up") or 0),
                "follow_ups": follow_ups,
            }
            _write_hdfs_json(f"{args.output}/summary.json", summary, spark)
        finally:
            spark.stop()


def _build_source_archive(runtime_dir: Path) -> Path:
    """打包项目源码供 Spark Python worker 反序列化模型时导入。"""
    source_root = Path(__file__).resolve().parents[1] / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"项目源码目录不存在: {source_root}")
    archive = runtime_dir / "cardiospark_src.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for source_file in source_root.rglob("*.py"):
            package.write(source_file, source_file.relative_to(source_root.parent))
    return archive


def _infer_coverage(frame, columns):
    province_values = _distinct_values(frame, "province", columns)
    city_values = _distinct_values(frame, "city", columns)
    region_values = _distinct_values(frame, "region", columns)
    district_values = _distinct_values(frame, "district", columns)
    community_field = "community_id" if "community_id" in columns else "community"
    community_values = _distinct_values(frame, community_field, columns)
    parent_values = city_values or region_values

    if len(district_values) > 1:
        is_chengdu = any("成都" in value for value in parent_values)
        return {
            "coverage_level": "district",
            "coverage_name": parent_values[0] if parent_values else "区域数据",
            "map_name": "chengdu" if is_chengdu else None,
            "map_available": is_chengdu,
            "region_field": "district",
            "region_count": len(district_values),
        }
    if len(community_values) > 1:
        return {
            "coverage_level": "community",
            "coverage_name": (district_values or parent_values or ["社区数据"])[0],
            "map_name": None,
            "map_available": False,
            "region_field": community_field,
            "region_count": len(community_values),
        }
    if len(district_values) == 1:
        return {
            "coverage_level": "table",
            "coverage_name": district_values[0],
            "map_name": None,
            "map_available": False,
            "region_field": "district",
            "region_count": 1,
        }
    if len(parent_values) > 1:
        return {
            "coverage_level": "city",
            "coverage_name": "多区域数据",
            "map_name": None,
            "map_available": False,
            "region_field": "city" if city_values else "region",
            "region_count": len(parent_values),
        }
    if province_values:
        return {
            "coverage_level": "province",
            "coverage_name": province_values[0],
            "map_name": None,
            "map_available": False,
            "region_field": "province",
            "region_count": len(province_values),
        }
    return {
        "coverage_level": "unknown",
        "coverage_name": "未识别区域",
        "map_name": None,
        "map_available": False,
        "region_field": None,
        "region_count": 0,
    }


def _resolve_field_mapping(columns):
    """在 Spark 作业中复用与本地分析一致的字段别名规则。"""
    from src.utils.field_mapping import resolve_field_mapping

    return resolve_field_mapping(columns)


def _distinct_values(frame, field, columns):
    if field not in columns:
        return []
    from pyspark.sql import functions as F

    rows = (
        frame.select(F.trim(F.col(field).cast("string")).alias("value"))
        .where(F.col("value").isNotNull() & (F.col("value") != ""))
        .distinct()
        .limit(1001)
        .collect()
    )
    return [str(row["value"]) for row in rows]


def _metric_expression(frame, target, labels_available, columns):
    from pyspark.sql import functions as F

    label = f"label_{target}"
    probability = f"{target}_probability"
    if labels_available and label in columns:
        return F.when(F.col(label).cast("double") == 1, 1).otherwise(0)
    if probability in columns:
        return F.when(F.col(probability) >= HIGH_RISK_THRESHOLD, 1).otherwise(0)
    return F.lit(0)


def _comorbidity_expression(frame, labels_available, columns):
    from pyspark.sql import functions as F

    if labels_available and {"label_heart", "label_stroke"}.issubset(columns):
        return F.when(
            (F.col("label_heart").cast("double") == 1)
            & (F.col("label_stroke").cast("double") == 1),
            1,
        ).otherwise(0)
    if {"heart_probability", "stroke_probability"}.issubset(columns):
        return F.when(
            (F.col("heart_probability") >= HIGH_RISK_THRESHOLD)
            & (F.col("stroke_probability") >= HIGH_RISK_THRESHOLD),
            1,
        ).otherwise(0)
    return F.lit(0)


def _region_rows(rows, coverage, min_group_size, metric_available):
    result = []
    for row in sorted(rows, key=lambda item: int(item["residents"]), reverse=True):
        residents = int(row["residents"])
        suppressed = residents < min_group_size
        denominator = residents or 1
        result.append(
            {
                "region": str(row[0]),
                "district": str(row[0]) if coverage["coverage_level"] == "district" else None,
                "community": str(row[0]) if coverage["coverage_level"] == "community" else None,
                "residents": None if suppressed else residents,
                "heart_rate": None if suppressed or not metric_available else _percent(row["heart_numerator"], denominator),
                "stroke_rate": None if suppressed or not metric_available else _percent(row["stroke_numerator"], denominator),
                "comorbidity_rate": None if suppressed or not metric_available else _percent(row["comorbidity_numerator"], denominator),
                "suppressed": suppressed,
            }
        )
    return result


def _age_groups(frame, labels_available, columns):
    from pyspark.sql import functions as F

    if "age_group" in columns:
        age_group = F.col("age_group").cast("string")
    else:
        age = F.col("age").cast("double")
        age_group = (
            F.when(age < 45, "18-44")
            .when(age < 55, "45-54")
            .when(age < 65, "55-64")
            .when(age < 75, "65-74")
            .otherwise("75+")
        )
    heart = _metric_expression(frame, "heart", labels_available, columns)
    stroke = _metric_expression(frame, "stroke", labels_available, columns)
    rows = frame.withColumn("_age_group", age_group).withColumn("_heart", heart).withColumn(
        "_stroke", stroke
    ).groupBy("_age_group").agg(
        F.count(F.lit(1)).alias("residents"),
        F.sum("_heart").alias("heart"),
        F.sum("_stroke").alias("stroke"),
    ).orderBy("_age_group").collect()
    return [
        {
            "age_group": str(row["_age_group"]),
            "residents": int(row["residents"]),
            "heart": int(row["heart"] or 0),
            "stroke": int(row["stroke"] or 0),
        }
        for row in rows
    ]


def _risk_factors(frame):
    from pyspark.sql import functions as F

    expressions = {
        "高血压": ("hypertension", lambda col: col == 1),
        "糖尿病": ("diabetes", lambda col: col == 1),
        "血脂异常": ("cholesterol", lambda col: col >= 2),
        "当前吸烟": ("smoker", lambda col: col == 2),
        "超重或肥胖": ("bmi", lambda col: col >= 25),
        "缺乏规律运动": ("exercise", lambda col: col == 0),
    }
    aggregations = []
    available = []
    for name, (field, predicate) in expressions.items():
        if field not in frame.columns:
            continue
        available.append(name)
        aggregations.append(
            F.sum(F.when(predicate(F.col(field).cast("double")), 1).otherwise(0)).alias(name)
        )
    if not aggregations:
        return []
    row = frame.agg(*aggregations).collect()[0]
    total = frame.count()
    return [
        {"name": name, "rate": _percent(row[name], total)}
        for name in sorted(available, key=lambda item: float(row[item] or 0), reverse=True)
    ]


def _follow_up_records(frame, prediction_enabled, columns):
    """输出有限的脱敏模型高风险名单，避免把居民明细返回浏览器。"""
    if not prediction_enabled or not {"heart_probability", "stroke_probability"}.issubset(columns):
        return []
    from pyspark.sql import functions as F

    selected = frame.where(
        (F.col("heart_probability") >= HIGH_RISK_THRESHOLD)
        | (F.col("stroke_probability") >= HIGH_RISK_THRESHOLD)
    ).withColumn(
        "_risk_score",
        F.greatest(F.col("heart_probability"), F.col("stroke_probability")) * 5,
    ).orderBy(F.desc("_risk_score")).limit(500)
    fields = [
        field for field in ("resident_id", "name", "phone", "age", "district", "community", "community_id")
        if field in columns
    ]
    fields += ["heart_probability", "stroke_probability", "_risk_score"]
    records = []
    for row in selected.select(*fields).collect():
        item = row.asDict()
        source_id = item.pop("resident_id", None)
        name = item.pop("name", None)
        phone = item.pop("phone", None)
        item["record_id"] = "R-" + hashlib.sha256(str(source_id or len(records)).encode("utf-8")).hexdigest()[:10]
        item["name"] = (str(name)[:1] + "**") if name else "居民"
        if phone:
            digits = re.sub(r"\D", "", str(phone))
            item["phone"] = digits[:3] + "****" + digits[-4:] if len(digits) >= 7 else "已脱敏"
        item["risk_score"] = round(float(item.pop("_risk_score", 0)), 3)
        item["heart_probability"] = round(float(item.get("heart_probability") or 0), 6)
        item["stroke_probability"] = round(float(item.get("stroke_probability") or 0), 6)
        item["screening_basis"] = "双模型概率达到高风险阈值"
        records.append(item)
    return records


def _invalid_row_count(frame):
    from pyspark.sql import functions as F

    age = F.col("age").cast("double")
    return frame.select(
        F.sum(F.when(age.isNull() | (age <= 0) | (age > 120), 1).otherwise(0)).alias("invalid")
    )


def _percent(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator) * 100, 2)


def _write_hdfs_json(path, payload, spark):
    """通过 Spark 使用 Hadoop 原生 FileSystem 写入 ADS 汇总结果。"""
    content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    jvm = spark._jvm
    jsc = spark.sparkContext._jsc
    configuration = jsc.hadoopConfiguration()
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    filesystem = hadoop_path.getFileSystem(configuration)
    stream = filesystem.create(hadoop_path, True)
    try:
        stream.write(bytearray(content))
    finally:
        stream.close()


def _score_with_dual_models(frame, manifest_path, columns, spark, score_engine="pandas"):
    """将本地模型分发到 Spark worker，返回评分后的 DataFrame。"""
    if not manifest_path:
        return frame, False, "unknown"
    try:
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest.get("models", {})
        paths = {}
        for target in ("heart", "stroke"):
            raw_path = models.get(target)
            if not raw_path:
                return frame, False, "unknown"
            artifact = Path(raw_path)
            if not artifact.is_absolute():
                artifact = manifest_path.parent / artifact
            artifact = artifact.resolve()
            if not artifact.exists():
                raise FileNotFoundError(f"模型文件不存在: {artifact}")
            paths[target] = artifact
        if not set(FEATURE_COLUMNS).issubset(columns):
            return frame, False, "unknown"

        # Spark distributes files under their original basenames. The two model names
        # are already distinct in the model registry, so worker lookup remains stable.
        heart_filename = paths["heart"].name
        stroke_filename = paths["stroke"].name
        spark.sparkContext.addFile(str(paths["heart"]), recursive=False)
        spark.sparkContext.addFile(str(paths["stroke"]), recursive=False)
        from pyspark.sql.types import DoubleType, StructField, StructType

        schema = StructType(list(frame.schema.fields) + [
            StructField("heart_probability", DoubleType(), True),
            StructField("stroke_probability", DoubleType(), True),
        ])
        model_version = manifest.get("updated_at", "unknown")
        model_cache = {}

        def load_models():
            import joblib
            from pyspark import Row as SparkRow, SparkFiles

            if not model_cache:
                model_cache["heart"] = joblib.load(SparkFiles.get(heart_filename))
                model_cache["stroke"] = joblib.load(SparkFiles.get(stroke_filename))
            return model_cache

        def score_batches(iterator):
            models = load_models()
            for batch in iterator:
                features = _prepare_model_features(batch.loc[:, FEATURE_COLUMNS])
                output = batch.copy()
                output["heart_probability"] = models["heart"].predict_proba(features)[:, 1]
                output["stroke_probability"] = models["stroke"].predict_proba(features)[:, 1]
                yield output

        if str(score_engine or "pandas").lower() != "pandas":
            raise ValueError("阶段二 Spark 评分固定使用 Pandas UDF。")
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401

        return frame.mapInPandas(score_batches, schema=schema), True, model_version
    except (OSError, ImportError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Spark 双模型评分初始化失败: {exc}") from exc


def _prepare_model_features(dataframe):
    import pandas as pd

    features = dataframe.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    features["age"] = features["age"].clip(18, 95).round().astype(int)
    features["bmi"] = features["bmi"].clip(10.3, 79.8).round(2)
    features["cholesterol"] = features["cholesterol"].clip(1, 3).round().astype(int)
    features["smoker"] = features["smoker"].clip(0, 2).round().astype(int)
    for field in ("gender", "diabetes", "hypertension", "alcohol", "exercise"):
        features[field] = features[field].clip(0, 1).round().astype(int)
    return features


if __name__ == "__main__":
    main()
