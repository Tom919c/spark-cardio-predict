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

阶段二新增链路：

- `src/dao/`：任务、数据集、分片和可选 HDFS 访问抽象。
- `src/services/analysis_task_service.py`：后台线程任务编排；本地模式执行分块 CSV 分析，HDFS 模式提交 Spark 作业。
- `src/services/phase2_analysis_service.py`：本地模式的分块分析实现，与 Spark ADS 输出保持相同结果契约。
- `data/localstorage/`：仅用于本地运行的 SQLite、上传临时文件和分析结果，不提交 Git。
- `static/geo/`、`static/echarts/`：可离线加载的地图和前端图表资源。
- `resources/knowledge/`：可审核的本地健康建议规则，不保存个人健康数据。

阶段一原型默认采用本地模式：当前成员在自己的环境中启动 Flask，直接读取 `data/raw/` 中的标准训练表和成都仿真表。`data/features/` 保存特征产物，`data/models/` 保存模型与清单。阶段二的 HDFS 模式由当前环境原生执行 Spark，WSL2 或 VMware Ubuntu 提供 Hadoop/HDFS 服务；作业顺序为 ETL -> 特征插补 -> 双模型评分 -> 群体聚合/重点筛查，所有 HDFS 参数通过 `.env` 配置。

阶段二上传链路在本地模式下使用 SQLite 和分块 CSV 分析，便于 Windows 开发和接口测试；在 HDFS 模式下，分片和原始文件进入 HDFS，后台线程提交 `scripts/run_phase2_analysis.py`，结果写入配置的 ADS 路径。Flask 只读取聚合结果，不将大文件完整拉回内存。

系统通过当前环境的 `SPARK_SUBMIT_BIN` 原生提交 `scripts/run_phase2_analysis.py`。Spark 读取 HDFS 原始数据，通过 `SparkContext.addFile` 分发双模型，并自动打包分发项目 `src` 包，保证自定义模型类可以在 worker 中反序列化；随后使用 `mapInPandas` 向量化评分。模型、Pandas、scikit-learn、joblib 和 pyarrow 必须在执行 Spark 的环境中可用。HDFS 的 WebHDFS 地址由 `HDFS_WEB_URL` 提供给上传服务和 Flask 读取 ADS 汇总结果，NameNode 地址由 `HDFS_NAMENODE_URI` 提供给 Spark。
