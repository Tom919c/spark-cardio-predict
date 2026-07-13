# 训练结果

阶段一双模型已在本地 Conda 环境完成正式训练；当前活跃模型以 `data/models/active_models.json` 为准。

| 模型 | 标签 | 指标 |
| --- | --- | --- |
| 心脏事件随机森林 | `label_heart` | ROC AUC 0.7342，Recall 0.1924 |
| 卒中随机森林 | `label_stroke` | ROC AUC 0.8328，Recall 0.7485 |

旧版逻辑回归、三分类和多轮随机种子结果不再作为当前版本依据。

本次 DWD 训练表未提供 `region` 字段，样本权重均为 `1.0`；若后续接入带来源字段的混合数据，需要重新训练并重新验收指标。

---

## Auto Training Record 2026-07-13 12:19:13
- Strategy: `balanced_subsample_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_phase1_20260713_121853.joblib`
- Accuracy: `0.8675`
- F1 Score: `0.2986`
- F2 Score: `0.2243`
- PR AUC: `0.3838`
- Recall: `0.1924`
- Specificity: `0.9834`
- ROC AUC: `0.7342`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_phase1_20260713_121913.joblib`
- Accuracy: `0.7709`
- F1 Score: `0.1192`
- F2 Score: `0.2405`
- PR AUC: `0.0827`
- Recall: `0.7485`
- Specificity: `0.7714`
- ROC AUC: `0.8328`
- Manifest Updated At: `2026-07-13T04:19:13.311069+00:00`
