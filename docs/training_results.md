# 模型训练结果记录

本文档用于记录项目中每次模型训练的输出结果，方便后续对比不同算法效果、选择默认模型、撰写项目报告。

建议后续每次训练至少记录以下信息：

1. 训练时间
2. 模型名称
3. 训练集大小
4. 测试集大小
5. Accuracy
6. F1 Score
7. ROC AUC
8. 模型保存路径
9. 备注

---

## 训练记录 1

- 模型名称：`logistic_regression`
- 训练集大小：`[54995, 13]`
- 测试集大小：`[13749, 13]`
- Accuracy：`0.7177`
- F1 Score：`0.7022`
- ROC AUC：`0.7784`
- 模型保存路径：`data/feature/models/logistic_regression.joblib`
- 备注：
  - 使用 `cardio_train.csv` 清洗后的数据训练
  - 作为第一版基线模型

### 分类报告摘要

- 类别 `0`
  - Precision：`0.7038`
  - Recall：`0.7619`
  - F1 Score：`0.7317`
  - Support：`6946`

- 类别 `1`
  - Precision：`0.7345`
  - Recall：`0.6726`
  - F1 Score：`0.7022`
  - Support：`6803`

---

## 训练记录 2

- 模型名称：`random_forest`
- 训练集大小：`[54995, 13]`
- 测试集大小：`[13749, 13]`
- Accuracy：`0.7350`
- F1 Score：`0.7193`
- ROC AUC：`0.7991`
- 模型保存路径：`data/feature/models/random_forest.joblib`
- 备注：
  - 使用 `cardio_train.csv` 清洗后的数据训练
  - 当前效果优于逻辑回归，可作为当前优先候选模型

### 分类报告摘要

- 类别 `0`
  - Precision：`0.7182`
  - Recall：`0.7825`
  - F1 Score：`0.7489`
  - Support：`6946`

- 类别 `1`
  - Precision：`0.7555`
  - Recall：`0.6865`
  - F1 Score：`0.7193`
  - Support：`6803`

---

## 当前阶段结论

- 已完成模型：
  - `logistic_regression`
  - `random_forest`

- 当前推荐模型：
  - `random_forest`

- 推荐原因：
  - Accuracy 更高
  - F1 Score 更高
  - ROC AUC 更高

---

## 自动训练记录 2026-07-10 11:40:08
- 模型名称：`random_forest`
- 训练轮数：`2`
- 实验标签：``
- 最优选择指标：`roc_auc`

### 每轮结果

#### 第 1 轮
- run_id：`random_forest_round1_20260710_114003`
- 随机种子：`42`
- 模型路径：`data\feature\models\random_forest_round1_20260710_114003.joblib`
- Accuracy：`0.735`
- F1 Score：`0.7193`
- ROC AUC：`0.7991`

#### 第 2 轮
- run_id：`random_forest_round2_20260710_114008`
- 随机种子：`43`
- 模型路径：`data\feature\models\random_forest_round2_20260710_114008.joblib`
- Accuracy：`0.7369`
- F1 Score：`0.7223`
- ROC AUC：`0.8023`

### 最终最佳结果
- 最佳轮次：`第 2 轮`
- 最佳模型路径：`data\feature\models\random_forest_round2_20260710_114008.joblib`
- 最终 Accuracy：`0.7369`
- 最终 F1 Score：`0.7223`
- 最终 ROC AUC：`0.8023`

---

## 自动训练记录 2026-07-10 11:40:52
- 模型名称：`random_forest`
- 训练轮数：`10`
- 实验标签：``
- 最优选择指标：`roc_auc`

### 每轮结果

#### 第 1 轮
- run_id：`random_forest_round1_20260710_114011`
- 随机种子：`42`
- 模型路径：`data\feature\models\random_forest_round1_20260710_114011.joblib`
- Accuracy：`0.735`
- F1 Score：`0.7193`
- ROC AUC：`0.7991`

#### 第 2 轮
- run_id：`random_forest_round2_20260710_114015`
- 随机种子：`43`
- 模型路径：`data\feature\models\random_forest_round2_20260710_114015.joblib`
- Accuracy：`0.7369`
- F1 Score：`0.7223`
- ROC AUC：`0.8023`

#### 第 3 轮
- run_id：`random_forest_round3_20260710_114020`
- 随机种子：`44`
- 模型路径：`data\feature\models\random_forest_round3_20260710_114020.joblib`
- Accuracy：`0.7358`
- F1 Score：`0.7201`
- ROC AUC：`0.8028`

#### 第 4 轮
- run_id：`random_forest_round4_20260710_114025`
- 随机种子：`45`
- 模型路径：`data\feature\models\random_forest_round4_20260710_114025.joblib`
- Accuracy：`0.7302`
- F1 Score：`0.7148`
- ROC AUC：`0.7948`

#### 第 5 轮
- run_id：`random_forest_round5_20260710_114029`
- 随机种子：`46`
- 模型路径：`data\feature\models\random_forest_round5_20260710_114029.joblib`
- Accuracy：`0.7315`
- F1 Score：`0.7165`
- ROC AUC：`0.7998`

#### 第 6 轮
- run_id：`random_forest_round6_20260710_114034`
- 随机种子：`47`
- 模型路径：`data\feature\models\random_forest_round6_20260710_114034.joblib`
- Accuracy：`0.7344`
- F1 Score：`0.7198`
- ROC AUC：`0.797`

#### 第 7 轮
- run_id：`random_forest_round7_20260710_114039`
- 随机种子：`48`
- 模型路径：`data\feature\models\random_forest_round7_20260710_114039.joblib`
- Accuracy：`0.7316`
- F1 Score：`0.7178`
- ROC AUC：`0.7953`

#### 第 8 轮
- run_id：`random_forest_round8_20260710_114043`
- 随机种子：`49`
- 模型路径：`data\feature\models\random_forest_round8_20260710_114043.joblib`
- Accuracy：`0.7355`
- F1 Score：`0.7186`
- ROC AUC：`0.8015`

#### 第 9 轮
- run_id：`random_forest_round9_20260710_114048`
- 随机种子：`50`
- 模型路径：`data\feature\models\random_forest_round9_20260710_114048.joblib`
- Accuracy：`0.7288`
- F1 Score：`0.7143`
- ROC AUC：`0.7966`

#### 第 10 轮
- run_id：`random_forest_round10_20260710_114052`
- 随机种子：`51`
- 模型路径：`data\feature\models\random_forest_round10_20260710_114052.joblib`
- Accuracy：`0.74`
- F1 Score：`0.7287`
- ROC AUC：`0.8043`

### 最终最佳结果
- 最佳轮次：`第 10 轮`
- 最佳模型路径：`data\feature\models\random_forest_round10_20260710_114052.joblib`
- 最终 Accuracy：`0.74`
- 最终 F1 Score：`0.7287`
- 最终 ROC AUC：`0.8043`
