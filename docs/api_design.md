# API

所有接口返回 `{ "success": boolean, "message": string, "data": object }`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/` | 机构端首页 |
| GET | `/health` | 服务健康检查 |
| GET | `/api/data/profile` | 当前训练数据配置 |
| GET | `/api/data/preview` | 训练数据预览 |
| GET | `/api/data/preprocess` | 特征工程校验结果 |
| GET | `/api/risk/summary` | 双模型状态 |
| GET | `/api/risk/train-estimate` | 训练耗时估算，不写入模型 |
| GET | `/api/risk/train?run_label=phase1` | 训练并注册两个模型 |
| POST | `/api/risk/predict` | 个人双风险评估 |
| GET | `/api/analysis/dashboard` | B 端群体指标 |
| GET | `/api/analysis/follow-ups?limit=100` | 重点随访名单 |
| POST | `/api/upload/tasks` | 创建 CSV 分片上传任务 |
| POST | `/api/upload/tasks/<task_id>/chunks` | 上传一个分片 |
| POST | `/api/upload/tasks/<task_id>/finalize` | 合并并提交后台分析 |
| GET | `/api/task/status/<task_id>` | 任务状态轮询 |
| GET | `/api/datasets/<dataset_id>/preview` | 脱敏预览数据集 |
| GET | `/api/datasets/<dataset_id>/result` | 获取群体分析结果 |
| POST | `/api/risk/what-if` | 双模型情景模拟 |
| POST | `/api/risk/batch` | 小批量双模型评估 |
| GET | `/api/trends/snapshots` | 获取历史快照 |
| POST | `/api/trends/compare` | 比较多个历史快照 |

`POST /api/risk/predict` 请求字段：`age`、`gender`、`bmi`、`cholesterol`、`diabetes`、`hypertension`、`smoker`、`alcohol`、`exercise`。浏览器不传模型路径。

模型尚未注册时，预测接口返回 HTTP 400 和明确的模型未就绪消息；浏览器不应自行生成预测结果。

`/api/analysis/follow-ups` 返回的是基于成都仿真数据标签的筛查名单，不是逐人模型概率；记录中的 `risk_score` 为 0~5 的筛查评分，`screening_basis` 会说明筛选规则。个人模型概率只由 `/api/risk/predict` 返回。

阶段二上传数据只用于质量检查、群体分析和明确请求的批量评估，不自动覆盖正式模型。机构页面只展示最近一次已上传且分析成功的数据集；没有成功上传数据时返回等待上传的初始化状态，不读取训练基线居民文件。大文件上传使用分片和后台任务，前端应轮询 `/api/task/status/<task_id>`，不能在浏览器中自行生成分析结果。社区人数少于配置的隐私阈值时，区域结果隐藏人数和比例。

机构上传文件按真实使用场景不要求疾病标签。正式模型可用时，机构风险率和高危名单统一使用双模型概率；即使上传文件意外包含标签列，标签也不能覆盖模型结果，只能作为离线校验字段。

上传任务必须填写 `data_period`，格式为 `YYYY-MM` 或 `YYYY-Qn`，用于历史快照比较。HDFS 模式下，当前运行环境原生调用 `SPARK_SUBMIT_BIN` 提交 Spark 作业，并通过 `HDFS_NAMENODE_URI` 和 `HDFS_WEB_URL` 访问 HDFS。每位成员只在自己的 `.env` 中填写本机路径、HDFS 地址和用户名，不得将个人路径提交到仓库。WSL2 与 VMware Ubuntu 使用相同的接口契约。
