# CVD_Standard_DWD 字段说明

默认训练数据：`data/raw/CVD_Standard_DWD_refined.csv`，由 `scripts/refine_training_dataset.py` 基于清洗后的 `CVD_Standard_DWD.csv` 重建，共 646,097 条记录。原始特征列保持不变，仅重建双模型训练标签，并生成同名 `_profile.json` 数据质量档案。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `age` | int | 年龄（岁） |
| `gender` | int | `0` 女，`1` 男 |
| `bmi` | float | 体质指数（kg/m²） |
| `cholesterol` | int | `1` 正常，`2` 偏高，`3` 高 |
| `diabetes` | int | `0` 否，`1` 是 |
| `hypertension` | int | `0` 否，`1` 是 |
| `smoker` | int | `0` 从不，`1` 曾吸烟，`2` 当前吸烟 |
| `alcohol` | int | `0` 否，`1` 是 |
| `exercise` | int | `0` 不规律，`1` 规律运动 |
| `label_heart` | int | 心脏风险筛查代理标签：`0` 否，`1` 是 |
| `label_stroke` | int | 脑卒中风险筛查代理标签：`0` 否，`1` 是 |

`label_heart` 与 `label_stroke` 独立保存，可同时为 `1`。它们由年龄、BMI、血脂、糖尿病、高血压、吸烟、饮酒和运动等特征的方向一致风险函数随机生成，分别控制为约 `3%` 与 `6%` 的训练正类比例。该口径用于课程项目的筛查风险建模，不等同于真实年度发病率、临床诊断或流行病学统计。系统分别训练两套 XGBoost 模型，不使用互斥三分类标签。

缺失特征在训练阶段使用 `IterativeImputer` 插补；标签列不参与插补。原始数据清洗过程中的详细字段映射、插补比例和来源说明保留在本地 `mydocs`，不进入 Git。

## DWS 仿真居民数据的标签口径

`data/raw/chengdu_resident_health_simulated.csv` 用于机构端群体分析，不替代 DWD 训练数据。该文件中的 `label_heart` 和 `label_stroke` 是依据《成都居民健康数据.md》年度新发率生成的事件代理标签：脑卒中约 `500.28/10万`，急性心肌梗死约 `79.36/10万`。它们不是模型筛查高危率，也不代表临床诊断结果。

机构端的“高危人群”单独依据事件代理标签或多项危险因素聚集规则筛选，不能把危险因素患病率、年度事件率和模型高风险率混为同一个百分比。
