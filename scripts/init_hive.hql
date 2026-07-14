-- CardioSpark Hive 外部表模板。
-- 使用前通过 hiveconf 覆盖路径，例如：
-- hive --hiveconf raw_path=/user/cardiospark/raw \
--      --hiveconf feature_path=/user/cardiospark/feature \
--      --hiveconf ads_path=/user/cardiospark/feature/ads -f scripts/init_hive.hql

CREATE DATABASE IF NOT EXISTS cardiospark;
USE cardiospark;

CREATE EXTERNAL TABLE IF NOT EXISTS ods_resident_health (
    age INT,
    gender INT,
    bmi DOUBLE,
    cholesterol INT,
    diabetes INT,
    hypertension INT,
    smoker INT,
    alcohol INT,
    exercise INT,
    resident_id STRING,
    region STRING,
    district STRING,
    community_id STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"')
STORED AS TEXTFILE
LOCATION '${hiveconf:raw_path}'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE IF NOT EXISTS dws_cardio_features (
    resident_id STRING,
    age INT,
    gender INT,
    bmi DOUBLE,
    cholesterol INT,
    diabetes INT,
    hypertension INT,
    smoker INT,
    alcohol INT,
    exercise INT,
    district STRING,
    community_id STRING
)
STORED AS PARQUET
LOCATION '${hiveconf:feature_path}';

CREATE EXTERNAL TABLE IF NOT EXISTS ads_region_risk_summary (
    region STRING,
    residents BIGINT,
    heart_rate DOUBLE,
    stroke_rate DOUBLE,
    comorbidity_rate DOUBLE,
    suppressed BOOLEAN
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"')
STORED AS TEXTFILE
LOCATION '${hiveconf:ads_path}'
TBLPROPERTIES ('skip.header.line.count'='1');
