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

---

## Auto Training Record 2026-07-13 20:35:34
- Strategy: `balanced_subsample_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_random_forest_round3_20260713_203458.joblib`
- Accuracy: `0.8675`
- F1 Score: `0.2986`
- F2 Score: `0.2243`
- PR AUC: `0.3838`
- Recall: `0.1924`
- Specificity: `0.9834`
- ROC AUC: `0.7342`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_random_forest_round3_20260713_203532.joblib`
- Accuracy: `0.7709`
- F1 Score: `0.1192`
- F2 Score: `0.2405`
- PR AUC: `0.0826`
- Recall: `0.7485`
- Specificity: `0.7714`
- ROC AUC: `0.8328`
- Manifest Updated At: `2026-07-13T12:35:34.040422+00:00`

---

## Auto Training Record 2026-07-13 20:46:37
- Strategy: `balanced_subsample_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_random_forest_round20_20260713_204620.joblib`
- Accuracy: `0.8675`
- F1 Score: `0.2986`
- F2 Score: `0.2243`
- PR AUC: `0.3838`
- Recall: `0.1924`
- Specificity: `0.9834`
- ROC AUC: `0.7342`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_random_forest_round20_20260713_204637.joblib`
- Accuracy: `0.7709`
- F1 Score: `0.1192`
- F2 Score: `0.2405`
- PR AUC: `0.0826`
- Recall: `0.7485`
- Specificity: `0.7714`
- ROC AUC: `0.8328`
- Manifest Updated At: `2026-07-13T12:46:37.921148+00:00`

---

## Auto Training Record 2026-07-13 21:13:21
- Strategy: `dual_torch_networks_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/neural_network_heart_nn_gpu_20260713_210720.joblib`
- Accuracy: `0.8661`
- F1 Score: `0.2882`
- F2 Score: `0.2159`
- PR AUC: `0.3731`
- Recall: `0.185`
- Specificity: `0.983`
- ROC AUC: `0.7165`

### stroke
- Run ID: ``
- Model Path: `data/models/neural_network_stroke_nn_gpu_20260713_211321.joblib`
- Accuracy: `0.7574`
- F1 Score: `0.1111`
- F2 Score: `0.2263`
- PR AUC: `0.0782`
- Recall: `0.7324`
- Specificity: `0.7579`
- ROC AUC: `0.8145`
- Manifest Updated At: `2026-07-13T13:13:21.282232+00:00`

---

## Auto Training Record 2026-07-13 21:13:50
- Strategy: `dual_torch_networks_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/neural_network_heart_nn_gpu_20260713_210823.joblib`
- Accuracy: `0.8668`
- F1 Score: `0.2885`
- F2 Score: `0.2154`
- PR AUC: `0.3709`
- Recall: `0.1843`
- Specificity: `0.984`
- ROC AUC: `0.7176`

### stroke
- Run ID: ``
- Model Path: `data/models/neural_network_stroke_nn_gpu_20260713_211350.joblib`
- Accuracy: `0.7515`
- F1 Score: `0.1111`
- F2 Score: `0.2273`
- PR AUC: `0.0757`
- Recall: `0.75`
- Specificity: `0.7516`
- ROC AUC: `0.8142`
- Manifest Updated At: `2026-07-13T13:13:50.333278+00:00`

---

## Auto Training Record 2026-07-13 21:28:24
- Strategy: `balanced_subsample_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_random_forest_20260713_212749.joblib`
- Accuracy: `0.8675`
- F1 Score: `0.2986`
- F2 Score: `0.2243`
- PR AUC: `0.3838`
- Recall: `0.1924`
- Specificity: `0.9834`
- ROC AUC: `0.7342`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_random_forest_20260713_212822.joblib`
- Accuracy: `0.7709`
- F1 Score: `0.1192`
- F2 Score: `0.2405`
- PR AUC: `0.0826`
- Recall: `0.7485`
- Specificity: `0.7714`
- ROC AUC: `0.8328`
- Manifest Updated At: `2026-07-13T13:28:24.768306+00:00`

---

## Auto Training Record 2026-07-13 21:28:27
- Strategy: `balanced_subsample_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_random_forest_20260713_212801.joblib`
- Accuracy: `0.8675`
- F1 Score: `0.2986`
- F2 Score: `0.2243`
- PR AUC: `0.3838`
- Recall: `0.1924`
- Specificity: `0.9834`
- ROC AUC: `0.7342`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_random_forest_20260713_212827.joblib`
- Accuracy: `0.7709`
- F1 Score: `0.1192`
- F2 Score: `0.2406`
- PR AUC: `0.0826`
- Recall: `0.7485`
- Specificity: `0.7714`
- ROC AUC: `0.8328`
- Manifest Updated At: `2026-07-13T13:28:27.730745+00:00`

---

## Auto Training Record 2026-07-13 21:35:20
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_20260713_213519.joblib`
- Accuracy: `0.8709`
- F1 Score: `0.3179`
- F2 Score: `0.2393`
- PR AUC: `0.4239`
- Recall: `0.2054`
- Specificity: `0.9851`
- ROC AUC: `0.7565`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_20260713_213520.joblib`
- Accuracy: `0.8005`
- F1 Score: `0.1301`
- F2 Score: `0.2559`
- PR AUC: `0.0849`
- Recall: `0.7205`
- Specificity: `0.8022`
- ROC AUC: `0.8463`
- Manifest Updated At: `2026-07-13T13:35:20.895918+00:00`

---

## Auto Training Record 2026-07-13 21:35:34
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_20260713_213531.joblib`
- Accuracy: `0.8709`
- F1 Score: `0.3179`
- F2 Score: `0.2393`
- PR AUC: `0.4239`
- Recall: `0.2054`
- Specificity: `0.9851`
- ROC AUC: `0.7565`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_20260713_213533.joblib`
- Accuracy: `0.8005`
- F1 Score: `0.1301`
- F2 Score: `0.2559`
- PR AUC: `0.0849`
- Recall: `0.7205`
- Specificity: `0.8022`
- ROC AUC: `0.8463`
- Manifest Updated At: `2026-07-13T13:35:34.000209+00:00`

---

## Auto Training Record 2026-07-14 15:04:55
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`
- Selected Rounds: `5`
- Best Round Index: `1`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_round1_20260714_150428.joblib`
- Accuracy: `0.8709`
- F1 Score: `0.3179`
- F2 Score: `0.2393`
- PR AUC: `0.4239`
- Recall: `0.2054`
- Specificity: `0.9851`
- ROC AUC: `0.7565`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_round1_20260714_150431.joblib`
- Accuracy: `0.8005`
- F1 Score: `0.1301`
- F2 Score: `0.2559`
- PR AUC: `0.0849`
- Recall: `0.7205`
- Specificity: `0.8022`
- ROC AUC: `0.8463`
- Manifest Updated At: `2026-07-14T07:04:55.451474+00:00`

---

## Auto Training Record 2026-07-14 15:05:01
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`
- Selected Rounds: `5`
- Best Round Index: `1`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_round1_20260714_150438.joblib`
- Accuracy: `0.8709`
- F1 Score: `0.3179`
- F2 Score: `0.2393`
- PR AUC: `0.4239`
- Recall: `0.2054`
- Specificity: `0.9851`
- ROC AUC: `0.7565`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_round1_20260714_150441.joblib`
- Accuracy: `0.8005`
- F1 Score: `0.1301`
- F2 Score: `0.2559`
- PR AUC: `0.0849`
- Recall: `0.7205`
- Specificity: `0.8022`
- ROC AUC: `0.8463`
- Manifest Updated At: `2026-07-14T07:05:01.516027+00:00`

---

## Auto Training Record 2026-07-14 15:21:52
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`
- Selected Rounds: `1`
- Best Round Index: `1`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_20260714_152150.joblib`
- Accuracy: `0.8703`
- F1 Score: `0.3358`
- F2 Score: `0.2582`
- PR AUC: `0.4166`
- Recall: `0.2238`
- Specificity: `0.9813`
- ROC AUC: `0.7536`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_20260714_152152.joblib`
- Accuracy: `0.8017`
- F1 Score: `0.1298`
- F2 Score: `0.255`
- PR AUC: `0.0848`
- Recall: `0.7141`
- Specificity: `0.8036`
- ROC AUC: `0.8462`
- Manifest Updated At: `2026-07-14T07:21:52.557077+00:00`

---

## Auto Training Record 2026-07-14 15:22:31
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`
- Selected Rounds: `1`
- Best Round Index: `1`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_20260714_152227.joblib`
- Accuracy: `0.8703`
- F1 Score: `0.3358`
- F2 Score: `0.2582`
- PR AUC: `0.4166`
- Recall: `0.2238`
- Specificity: `0.9813`
- ROC AUC: `0.7536`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_20260714_152231.joblib`
- Accuracy: `0.8017`
- F1 Score: `0.1298`
- F2 Score: `0.255`
- PR AUC: `0.0848`
- Recall: `0.7141`
- Specificity: `0.8036`
- ROC AUC: `0.8462`
- Manifest Updated At: `2026-07-14T07:22:31.332992+00:00`

---

## Auto Training Record 2026-07-14 15:22:34
- Strategy: `xgboost_gpu_with_train_only_minority_oversampling_and_isotonic_calibration`
- Selected Rounds: `1`
- Best Round Index: `1`

### heart
- Run ID: ``
- Model Path: `data/models/random_forest_heart_xgboost_gpu_20260714_152231.joblib`
- Accuracy: `0.8703`
- F1 Score: `0.3358`
- F2 Score: `0.2582`
- PR AUC: `0.4166`
- Recall: `0.2238`
- Specificity: `0.9813`
- ROC AUC: `0.7536`

### stroke
- Run ID: ``
- Model Path: `data/models/random_forest_stroke_xgboost_gpu_20260714_152234.joblib`
- Accuracy: `0.8017`
- F1 Score: `0.1298`
- F2 Score: `0.255`
- PR AUC: `0.0848`
- Recall: `0.7141`
- Specificity: `0.8036`
- ROC AUC: `0.8462`
- Manifest Updated At: `2026-07-14T07:22:34.039541+00:00`
