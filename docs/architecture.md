# 架构

```text
ODS 原始数据 -> DWD 标准双标签表 -> DWS 成都市仿真人口 -> ADS 模型与分析结果
                                  |                         |
                            Spark 伪分布式作业          Flask B/C 端接口
```

- `data/ods/source_dataset/`：6 个原始异构数据集。
- `data/dwd/`：`CVD_Standard_DWD.csv`，9 个特征与两个独立标签。
- `data/dws/`：200 万条成都市民仿真数据，供 B 端群体分析使用。
- `data/ads/models/`：训练后的两个模型与模型清单。
- `src/services/`：数据、风险、群体分析业务服务。
- `src/spark_jobs/`：ETL、DWS、群体聚合和高危筛查作业。

阶段一原型默认采用本地模式：Windows 使用 Conda `bigdata` 启动 Flask，直接读取 `data/dwd/` 和 `data/dws/`。WSL2 或 VMware Linux 只负责执行 Hadoop/Spark 伪分布式链路；作业顺序为 ETL -> 特征插补 -> 群体聚合/重点筛查，所有 HDFS 参数通过 `.env` 配置。
