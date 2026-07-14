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
| POST | `/api/risk/train` | 训练并注册两个模型，JSON 可传 `run_label`、`rounds` |
| POST | `/api/risk/predict` | 个人双风险评估 |
| GET | `/api/assessments` | 按 `X-Client-ID` 获取匿名个人评估历史 |
| GET | `/api/assessments/<assessment_id>` | 获取当前匿名客户端的一次评估摘要 |
| GET | `/api/capabilities` | 查看 SQLite/MySQL、HDFS、Spark 与模型就绪状态 |
| GET | `/api/analysis/dashboard` | B 端群体指标 |
| GET | `/api/analysis/follow-ups?limit=100` | 重点随访名单 |

`POST /api/risk/predict` 请求字段：`age`、`gender`、`bmi`、`cholesterol`、`diabetes`、`hypertension`、`smoker`、`alcohol`、`exercise`。浏览器不传模型路径。

页面可同时提交 `systolic_bp`、`diastolic_bp`、`fasting_glucose` 和
`family_history` 作为辅助健康提示字段；这些字段不进入当前九特征概率模型。
预测接口返回的是课程项目中的筛查代理风险，不是临床诊断或真实年度发病概率。

模型尚未注册时，预测接口返回 HTTP 400 和明确的模型未就绪消息；浏览器不应自行生成预测结果。

`/api/analysis/follow-ups` 返回的是基于成都仿真数据标签的筛查名单，不是逐人模型概率；记录中的 `risk_score` 为 0~5 的筛查评分，`screening_basis` 会说明筛选规则。个人模型概率只由 `/api/risk/predict` 返回。
