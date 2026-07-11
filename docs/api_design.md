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
| GET | `/api/risk/train?run_label=phase1` | 训练并注册两个模型 |
| POST | `/api/risk/predict` | 个人双风险评估 |
| GET | `/api/analysis/dashboard` | B 端群体指标 |
| GET | `/api/analysis/follow-ups?limit=100` | 重点随访名单 |

`POST /api/risk/predict` 请求字段：`age`、`gender`、`bmi`、`cholesterol`、`diabetes`、`hypertension`、`smoker`、`alcohol`、`exercise`。浏览器不传模型路径。

模型尚未注册时，预测接口返回 HTTP 400 和明确的模型未就绪消息；浏览器不应自行生成预测结果。
