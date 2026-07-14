# 架构

```text
浏览器/机构 CSV -> Flask 分片接口 -> HDFS ODS/RAW
                                      |
                                  Spark ETL
                                      |
                               DWS 特征 + 双模型评分
                                      |
                               HDFS ADS 聚合结果
                                      |
                     Flask API / 机构看板 / 趋势与随访

SQLite（默认）或 MySQL 保存任务、数据集、评估历史和模型/作业元数据；
HDFS 保存大文件与分布式结果，Spark 负责批量计算。三者职责相互独立。
```

- `data/raw/source_dataset/`：原始异构数据集。
- `data/raw/CVD_Standard_DWD.csv`：9 个特征与两个独立标签的标准数据。
- `data/raw/chengdu_resident_health_simulated.csv`：成都市民仿真数据，供 B 端群体分析使用。
- `data/models/`：训练后的两个模型与模型清单。
- `data/features/`：Spark 或本地特征产物。
- `src/services/`：数据、风险、群体分析业务服务。
- `src/spark_jobs/`：ETL、DWS、群体聚合和高危筛查作业。

阶段一原型默认采用本地模式：Windows 使用成员自行配置的 Conda 环境启动 Flask，直接读取 `data/raw/` 中的标准训练表和成都仿真表。`data/features/` 保存特征产物，`data/models/` 保存模型与清单。WSL2 或 VMware Linux 只负责执行 Hadoop/Spark 伪分布式链路；作业顺序为 ETL -> 特征插补 -> 群体聚合/重点筛查，所有 HDFS 参数通过 `.env` 配置。
