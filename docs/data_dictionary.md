# CVD_Standard_DWD 字段说明

默认训练数据：`data/raw/CVD_Standard_DWD.csv`，共 646,097 条记录、11 个字段。数据由 `data/raw/source_dataset/` 的异构原始数据集清洗、对齐和插补得到。

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
| `label_heart` | int | 心脏事件标签：`0` 否，`1` 是 |
| `label_stroke` | int | 脑卒中标签：`0` 否，`1` 是 |

`label_heart` 与 `label_stroke` 独立保存，可同时为 `1`。阶段一分别训练两套随机森林模型，不再使用互斥三分类标签。

缺失特征在训练阶段使用 `IterativeImputer` 插补；标签列不参与插补。原始数据清洗过程中的详细字段映射、插补比例和来源说明保留在本地 `mydocs`，不进入 Git。
