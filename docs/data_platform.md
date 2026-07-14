# 数据平台运行架构

CardioSpark 将“业务元数据”和“大规模健康数据”分开管理。该边界既保证课程演示可以开箱运行，也保留了 Hadoop/Spark 的真实处理链路。

## 1. 元数据库：SQLite 默认，MySQL 可切换

任务、上传分片、数据集登记、分析状态和匿名个人评估历史属于小规模结构化元数据。默认配置为：

```dotenv
DATABASE_TYPE=sqlite
DATABASE_PATH=data/localstorage/app.db
```

SQLite 不需要额外服务，适合单机开发和答辩。部署到多人或长期运行环境时，只需安装 MySQL 依赖、创建数据库并修改配置：

```dotenv
DATABASE_TYPE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=cardiospark
MYSQL_PASSWORD=<password>
MYSQL_DATABASE=cardiospark
```

Service 和 Controller 均通过数据库工厂访问统一 DAO 契约，因此业务层不需要改代码。SQLite 与 MySQL 只保存元数据，不保存完整居民健康明细。

## 2. 本地数据链路

`DATA_MODE=local` 时，服务端强制采用以下拓扑：

```text
CSV 分片 -> 本地文件系统 -> 分块 Pandas 分析 -> 本地 ADS JSON/CSV
              |                       |
              +---- SQLite/MySQL 任务状态 ----+
```

前端提交的 `use_hdfs` 字段不会改变部署模式，避免浏览器造成存储位置和计算引擎不一致。

## 3. Hadoop + Spark 分布式链路

`DATA_MODE=hdfs` 时，服务端强制执行真实的 HDFS/Spark 链路：

```text
CSV 分片 -> WebHDFS raw -> HDFS 合并原始文件
                              |
                              v
                    spark-submit ETL/校验
                              |
                    特征处理 + 双模型批量评分
                              |
                              v
              HDFS feature/ADS（汇总、区域指标、重点筛查）
                              |
                              v
                  Flask 只读取 ADS 聚合结果
```

核心配置：

```dotenv
DATA_MODE=hdfs
HDFS_NAMENODE_URI=hdfs://localhost:9000
HDFS_WEB_URL=http://localhost:9870
HDFS_RAW_PATH=/user/<user>/cardiospark/raw
HDFS_FEATURE_PATH=/user/<user>/cardiospark/feature
HDFS_RESULT_ROOT=/user/<user>/cardiospark/feature/ads
SPARK_SUBMIT_BIN=spark-submit
SPARK_SCORE_ENGINE=pandas
```

上传由 WebHDFS 完成；后台分析任务调用 `scripts/run_phase2_analysis.py`。Spark 负责 CSV ETL、数据质量校验、缺失值处理、双模型批量评分、区域聚合和 ADS 输出。Flask 不把居民级大表整体拉回单机内存。

## 4. 运行能力检查

启动 Flask 后访问：

```http
GET /api/capabilities
```

接口返回：

- 当前 `database.type` 以及 SQLite/MySQL 支持范围；
- 当前 `data_mode`、实际 `storage_engine` 和 `compute_engine`；
- HDFS 客户端是否可创建、NameNode/WebHDFS 是否已配置；
- 当前系统是否能发现 `hadoop` 与 `hdfs` 命令行；
- `spark-submit` 是否可执行、分析脚本是否存在；
- 心脏事件与脑卒中模型是否就绪；
- `distributed_infrastructure_ready`（HDFS 客户端与 Spark 提交条件）和包含双模型状态的 `distributed_pipeline_ready` 综合状态。

其中 `hdfs.client_ready` 只表示客户端依赖与配置能够创建，不代表远端集群网络一定连通。正式运行前仍应通过 Hadoop 管理命令或一次小文件上传验证 NameNode/DataNode。

Windows 环境若执行 `hadoop version` 时出现 Log4j 无法写入
`%HADOOP_HOME%/logs`，应先创建 logs 目录并确认当前用户有写权限；这属于本机
Hadoop 环境问题，不应在 Flask 中静默忽略。VMware/WSL2 环境则应分别检查
`hdfs dfs -ls /` 与 WebHDFS 9870 端口。

任务状态接口也会返回 `storage_engine` 和 `compute_engine`，这两个字段由服务层根据任务存储模式动态组装，不增加数据库字段。
