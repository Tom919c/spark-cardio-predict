# CVD_Standard_DWD Data Dictionary

Current project now targets the standardized `CVD_Standard_DWD` dataset and uses a dual-model strategy.

## Raw fields

`age`
Age in years.

`gender`
Binary encoded gender, `0/1`.

`bmi`
Body mass index.

`cholesterol`
Ordered category, valid values: `1/2/3`.

`diabetes`
Binary flag, `0/1`.

`hypertension`
Binary flag, `0/1`.

`smoker`
Three-level smoking status:
- `0` = never smoked
- `1` = formerly smoked
- `2` = currently smoking

`alcohol`
Binary flag, `0/1`.

`exercise`
Binary flag, `0/1`.

`target_disease`
Three-class disease label:
- `0` = healthy
- `1` = heart adverse event
- `2` = stroke

## Training labels used by the project

The project no longer uses a single final label directly for prediction output. It derives two binary targets:

`heart_risk`
- `0` = no heart adverse event
- `1` = heart adverse event
- derived as `target_disease == 1`

`stroke_risk`
- `0` = no stroke risk
- `1` = stroke risk
- derived as `target_disease == 2`

## Final four-category output

The final user-facing category is built by combining the two model outputs:

- `0` = healthy
- `1` = heart-only risk
- `2` = stroke-only risk
- `3` = both heart and stroke risk

This final four-category output is model-combined output, not a direct raw label column from the dataset.
