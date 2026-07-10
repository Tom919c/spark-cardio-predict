# 项目测试说明

本文档用于整理当前项目开发过程中用到的测试网址、命令和用途说明，方便后续直接按步骤验证功能。

---

## 一、启动项目

### 启动命令

```powershell
C:\Users\1\AppData\Local\Programs\Python\Python311\python.exe -m flask run
```

### 作用

- 启动 Flask 本地开发服务
- 让后续所有接口可通过 `http://127.0.0.1:5000` 访问

### 你需要关注什么

- 是否出现 `Running on http://127.0.0.1:5000`
- 如果改过代码，是否已经重启过服务

---

## 二、基础服务测试

### 1. 根路径

网址：

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

作用：

- 检查项目基础骨架是否启动成功
- 验证 Flask 路由注册是否正常

你需要关注什么：

- 是否返回 `success: true`
- 是否提示平台后端骨架已启动

### 2. 健康检查

网址：

[http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

作用：

- 检查 Flask 服务是否存活
- 用于排查“服务挂了”还是“业务接口有问题”

你需要关注什么：

- 是否返回“服务运行正常”

---

## 三、数据集相关测试

### 1. 数据集配置检查

网址：

[http://127.0.0.1:5000/api/data/profile](http://127.0.0.1:5000/api/data/profile)

作用：

- 检查数据集路径是否配置正确
- 检查文件是否存在
- 检查编码、分隔符、特征列和标签列配置

你需要关注什么：

- `configured`
- `exists`
- `dataset_file_path`
- `encoding`
- `separator`

### 2. 数据预览

网址：

[http://127.0.0.1:5000/api/data/preview](http://127.0.0.1:5000/api/data/preview)

作用：

- 预览 CSV 前几行内容
- 检查字段是否被正确拆分
- 检查数据规模是否正确

你需要关注什么：

- `columns`
- `shape`
- `preview_rows`

### 3. 数据预处理检查

网址：

[http://127.0.0.1:5000/api/data/preprocess](http://127.0.0.1:5000/api/data/preprocess)

作用：

- 检查训练前数据清洗、字段校验和衍生特征是否正常

你需要关注什么：

- `valid`
- `missing_columns`
- `processed_columns`
- `processed_shape`
- `train_shape`
- `test_shape`

---

## 四、风险训练模块测试

### 1. 风险平台摘要

网址：

[http://127.0.0.1:5000/api/risk/summary](http://127.0.0.1:5000/api/risk/summary)

作用：

- 检查风险模块配置是否加载成功
- 查看默认模型、支持模型、多轮训练配置

你需要关注什么：

- `default_model_name`
- `available_model_names`
- `multi_run_training`
- `default_training_rounds`
- `best_model_metric`

### 2. 单轮或默认轮数训练

网址示例：

[http://127.0.0.1:5000/api/risk/train?model_name=logistic_regression](http://127.0.0.1:5000/api/risk/train?model_name=logistic_regression)

作用：

- 触发模型训练
- 查看训练结果与评估指标

你需要关注什么：

- `model_name`
- `metrics`
- `model_path`

### 3. 指定多轮训练

网址示例：

[http://127.0.0.1:5000/api/risk/train?model_name=random_forest&rounds=2](http://127.0.0.1:5000/api/risk/train?model_name=random_forest&rounds=2)

网址示例：

[http://127.0.0.1:5000/api/risk/train?model_name=random_forest&rounds=10](http://127.0.0.1:5000/api/risk/train?model_name=random_forest&rounds=10)

作用：

- 一次性自动跑完指定轮数
- 比较每轮结果
- 自动选出最优模型

你需要关注什么：

- `rounds`
- `all_round_results`
- `best_result`
- `best_metric`

### 4. 模型文件检查

PowerShell 命令：

```powershell
Get-ChildItem C:\Users\1\PycharmProjects\FlaskProject1\data\feature\models
```

作用：

- 查看训练完成后实际生成了哪些 `.joblib` 模型文件
- 确认多轮训练是否真的保存了多轮模型

你需要关注什么：

- 文件名是否包含 `round1`、`round2` 等轮次信息
- 是否生成了你本次训练对应的新模型文件

---

## 五、单人风险预测测试

### 1. PowerShell 预测命令

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/risk/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "model_path": "C:/Users/1/PycharmProjects/FlaskProject1/data/feature/models/random_forest_round10_20260710_114052.joblib",
    "age": 18393,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }'
```

作用：

- 向单人风险预测接口发送一个人的健康指标
- 使用指定模型文件进行预测

你需要关注什么：

- `predicted_label`
- `predicted_probability`
- `risk_level`
- `model_path`

### 2. 展开完整预测结果

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/risk/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "model_path": "C:/Users/1/PycharmProjects/FlaskProject1/data/feature/models/random_forest_round10_20260710_114052.joblib",
    "age": 18393,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }'

$response | ConvertTo-Json -Depth 10
```

作用：

- 把 PowerShell 默认折叠的对象完整展开
- 便于查看中文风险说明、指标解读和干预建议

你需要关注什么：

- `risk_summary`
- `risk_probability_percent`
- `key_highlights`
- `indicator_insights`
- `intervention_plan`

### 3. 只查看重点字段

```powershell
$response.data.predicted_label
$response.data.predicted_probability
$response.data.risk_level
$response.data.risk_summary
$response.data.key_highlights
```

作用：

- 快速查看最重要的预测结果
- 避免完整 JSON 太长不方便阅读

---

## 六、训练结果文档检查

### 1. 训练结果记录

文档：

[training_results.md](C:/Users/1/PycharmProjects/FlaskProject1/docs/training_results.md)

作用：

- 查看每次训练的记录
- 查看多轮训练每轮结果和最终最佳结果是否已自动写入

你需要关注什么：

- 是否自动新增训练记录
- 是否包含每轮结果
- 是否包含最终最佳模型信息

### 2. 修改说明文档

文档：

[change_notes.md](C:/Users/1/PycharmProjects/FlaskProject1/docs/change_notes.md)

作用：

- 查看每个阶段的代码修改目的、文件作用和函数职责

你需要关注什么：

- 是否有新增阶段说明
- 是否有本次功能的解释记录

---

## 七、常见问题排查

### 1. 服务能开，但接口访问失败

先检查：

[http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

作用：

- 判断是不是整个 Flask 服务挂了

### 2. 预测时报模型文件不存在

先执行：

```powershell
Get-ChildItem C:\Users\1\PycharmProjects\FlaskProject1\data\feature\models
```

作用：

- 查出真实存在的模型文件名
- 把正确的模型路径填进 `model_path`

### 3. 返回结果太长看不清

执行：

```powershell
$response | ConvertTo-Json -Depth 10
```

作用：

- 展开完整 JSON

### 4. 改了代码但结果没变化

说明：

- 当前如果不是 Debug 自动重载模式，改完代码要手动重启 Flask 服务

---

## 八、建议测试顺序

推荐你每次按这个顺序检查：

1. `/health`
2. `/api/data/profile`
3. `/api/data/preview`
4. `/api/data/preprocess`
5. `/api/risk/summary`
6. `/api/risk/train?...`
7. 模型目录
8. `/api/risk/predict`
9. `training_results.md`
