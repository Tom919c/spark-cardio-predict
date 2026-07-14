# 答辩证据清单

本目录清单用于展示 CardioSpark 从数据治理、基线模型、模型优化到
Hadoop/Spark 批处理和最终交付的开发过程。候选模型二进制已清理，关键指标与
可复现实验脚本保留。

## 1. 数据治理

- `data/raw/CVD_Standard_DWD.csv`：统一后的 DWD 输入。
- `data/raw/CVD_Standard_DWD_refined.csv`：最终训练表。
- `data/raw/CVD_Standard_DWD_refined_profile.json`：标签语义、阳性率、年龄分层与随机种子。
- `scripts/refine_training_dataset.py`：筛查代理标签的可复现生成规则。
- `scripts/generate_chengdu_health_data.py`：成都仿真人口生成规则。

## 2. 模型演进

- `data/model_benchmark_baseline.json`：逻辑回归、随机森林、ExtraTrees、梯度提升等基线。
- `data/model_benchmark_with_xgboost.json`：加入 XGBoost 后的对比。
- `data/model_benchmark_clinical_stress.json`：典型风险情景压力测试。
- `data/xgboost_hyperparameter_search.json`：关键超参数搜索结果。
- `data/xgboost_phase2_v2_training.json`：最终候选训练摘要。
- `docs/training_results.md`：按时间记录的正式训练指标。
- `data/models/active_models.json`：最终模型契约、依赖版本、文件哈希和测试集指标。
- `data/models/active_heart.joblib`、`active_stroke.joblib`：最终交付模型。

模型标签属于课程项目中的风险筛查代理标签。上述指标用于比较模型复现筛查规则
和风险排序的能力，不是外部临床验证结果。

## 3. SQLite / MySQL 元数据层

- `data/localstorage/app.db`：本地演示中的上传任务、数据集和匿名评估历史。
- `src/dao/database_dao.py`：SQLite DAO。
- `src/dao/mysql_dao.py`：相同业务契约的 MySQL DAO。
- `scripts/init_mysql.sql`：MySQL 建表脚本。

答辩时可通过 `GET /api/assessments` 展示个人评估历史持久化，通过
`GET /api/task/status/<task_id>` 展示分片与分析任务状态。

## 4. Hadoop / Spark 计算链路

- `scripts/upload_to_hdfs.py`：HDFS 数据上传。
- `scripts/run_phase2_analysis.py`：Spark ETL、Pandas UDF 双模型评分、区域聚合与 ADS 输出。
- `src/services/analysis_task_service.py`：`spark-submit` 任务编排与状态回写。
- `scripts/init_hive.hql`：ODS、DWS、ADS 外部表模板。
- `data/localstorage/results/*/summary.json`：与 Spark ADS 契约一致的本地结果样例。
- `docs/data_platform.md`：SQLite/MySQL、HDFS、Spark 的完整拓扑与切换方式。

答辩前访问 `GET /api/capabilities`，确认当前环境中的数据库、HDFS、Spark、分析
脚本与双模型是否就绪。HDFS 环境不可用时，可用本地模式展示相同结果契约，但应
明确说明当前演示所使用的计算引擎。

## 5. 推荐演示顺序

1. 展示数据质量档案和代理标签边界。
2. 展示基线模型与 XGBoost 对比、最终阈值与召回率。
3. 完成一次个人评估，展示 SQLite 匿名历史与结构化健康行动建议。
4. 上传一个带时期的机构 CSV，展示任务状态中的存储与计算引擎。
5. 在 Hadoop 环境展示 HDFS 路径、Spark 作业日志和 ADS 汇总结果。
6. 通过 `/api/capabilities` 总结当前部署真正启用的技术组件。
