# cardio_train.csv 数据字段说明

当前项目第一阶段默认使用本地 `cardio_train.csv` 数据集，不再接外部采集接口。

## 推荐原始字段

`id`
数据主键。

`age`
年龄，公开心血管数据集中通常以“天”为单位存储。

`gender`
性别编码。

`height`
身高，通常单位为厘米。

`weight`
体重，通常单位为千克。

`ap_hi`
收缩压。

`ap_lo`
舒张压。

`cholesterol`
胆固醇等级。

`gluc`
血糖等级。

`smoke`
是否吸烟。

`alco`
是否饮酒。

`active`
是否保持运动。

`cardio`
是否存在心血管疾病风险标签，作为监督学习目标列。

## 第一阶段衍生字段

`age_years`
由 `age` 转换得到，单位为年。

`bmi`
由 `height` 与 `weight` 计算得到的身体质量指数。

## 当前仍需你确认的内容

- `DATASET_FILE_PATH`
  建议填写为项目内路径，或你确认的真实本地路径。
- `DATASET_SEPARATOR`
  公开 `cardio_train.csv` 常见分隔符是 `;`，如果你的文件不是这个格式，需要改这里。
- `DATASET_ENCODING`
  默认按 `utf-8` 处理，如果读取报错，需要你确认真实编码。
