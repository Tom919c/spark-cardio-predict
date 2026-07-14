# 训练结果

阶段一双模型已在本地 Conda 环境完成正式训练；当前活跃模型以 `data/models/active_models.json` 为准。

| 模型 | 标签 | 指标 |
| --- | --- | --- |
| 心脏事件 XGBoost | `label_heart` | ROC AUC 0.8810，Recall 0.5587，Precision 0.2180 |
| 卒中 XGBoost | `label_stroke` | ROC AUC 0.8781，Recall 0.7125，Precision 0.2497 |

旧版随机森林、候选 XGBoost 和调参结果只用于开发过程对比，不再作为当前版本依据。
候选模型二进制已在结题清理中删除，关键指标 JSON 和以下历史训练记录保留用于答辩。

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

---

## Auto Training Record 2026-07-14 11:35:15
- Strategy: `xgboost_with_natural_prevalence_training_isotonic_calibration_and_threshold_tuning`

### heart
- Run ID: ``
- Model Path: `data/models/xgboost_heart_xgboost_refined_data_candidate_20260714_113155.joblib`
- Accuracy: `0.9269`
- F1 Score: `0.3146`
- F2 Score: `0.4295`
- PR AUC: `0.2715`
- Recall: `0.5679`
- Specificity: `0.9378`
- ROC AUC: `0.8811`

### stroke
- Run ID: ``
- Model Path: `data/models/xgboost_stroke_xgboost_refined_data_candidate_20260714_113200.joblib`
- Accuracy: `0.8599`
- F1 Score: `0.3738`
- F2 Score: `0.5172`
- PR AUC: `0.3916`
- Recall: `0.695`
- Specificity: `0.8705`
- ROC AUC: `0.8766`
- Manifest Updated At: `2026-07-14T03:35:15.788824+00:00`

---

## Auto Training Record 2026-07-14 12:04:25
- Strategy: `xgboost_with_natural_prevalence_training_isotonic_calibration_and_threshold_tuning`

### heart
- Run ID: ``
- Model Path: `data/models/xgboost_heart_xgboost_phase2_v2_20260714_120246.joblib`
- Accuracy: `0.9278`
- F1 Score: `0.3136`
- F2 Score: `0.4257`
- PR AUC: `0.2697`
- Recall: `0.5587`
- Specificity: `0.939`
- ROC AUC: `0.881`

### stroke
- Run ID: ``
- Model Path: `data/models/xgboost_stroke_xgboost_phase2_v2_20260714_120254.joblib`
- Accuracy: `0.8539`
- F1 Score: `0.3697`
- F2 Score: `0.5198`
- PR AUC: `0.3987`
- Recall: `0.7125`
- Specificity: `0.8629`
- ROC AUC: `0.8781`
- Manifest Updated At: `2026-07-14T04:04:25.589352+00:00`

---

## Auto Training Record 2026-07-14 19:58:00
- Strategy: `xgboost_with_natural_prevalence_training_isotonic_calibration_and_threshold_tuning`

### heart
- Run ID: ``
- Model Path: `data/models/active_heart.joblib`
- Accuracy: `0.9278`
- F1 Score: `0.3136`
- F2 Score: `0.4257`
- PR AUC: `0.2697`
- Recall: `0.5587`
- Specificity: `0.939`
- ROC AUC: `0.881`

### stroke
- Run ID: ``
- Model Path: `data/models/active_stroke.joblib`
- Accuracy: `0.8539`
- F1 Score: `0.3697`
- F2 Score: `0.5198`
- PR AUC: `0.3987`
- Recall: `0.7125`
- Specificity: `0.8629`
- ROC AUC: `0.8781`
- Manifest Updated At: `2026-07-14T11:58:00.349728+00:00`

---

## Auto Training Record 2026-07-14 21:24:35
- Strategy: `xgboost_with_natural_prevalence_training_isotonic_calibration_and_threshold_tuning`

### heart
- Run ID: ``
- Model Path: `data/models/active_heart.joblib`
- Accuracy: `0.9278`
- F1 Score: `0.3136`
- F2 Score: `0.4257`
- PR AUC: `0.2697`
- Recall: `0.5587`
- Specificity: `0.939`
- ROC AUC: `0.881`

### stroke
- Run ID: ``
- Model Path: `data/models/active_stroke.joblib`
- Accuracy: `0.8539`
- F1 Score: `0.3697`
- F2 Score: `0.5198`
- PR AUC: `0.3987`
- Recall: `0.7125`
- Specificity: `0.8629`
- ROC AUC: `0.8781`
- Manifest Updated At: `2026-07-14T13:24:35.374461+00:00`
