import pandas as pd

from src.ml.feature_engineering import CardioFeatureEngineering


FEATURES = [
    "age", "gender", "bmi", "cholesterol", "diabetes", "hypertension",
    "smoker", "alcohol", "exercise",
]


def test_build_training_dataframe_imputes_features_and_keeps_two_labels():
    dataframe = pd.DataFrame(
        [
            [55, 1, 26.1, 2, 0, 1, 2, 1, 0, 1, 0],
            [64, 0, None, 3, 1, 1, 0, 0, 1, 0, 1],
            [48, 1, 22.4, 1, 0, 0, 0, 0, 1, 0, 0],
            [70, 0, 31.0, 3, 1, 1, 1, 1, 0, 1, 1],
        ],
        columns=FEATURES + ["label_heart", "label_stroke"],
    )

    result = CardioFeatureEngineering(FEATURES, "target_disease", 0.2, 42).build_training_dataframe(dataframe)

    assert result[FEATURES].isna().sum().sum() == 0
    assert result["label_heart"].tolist() == [1, 0, 0, 1]
    assert result["label_stroke"].tolist() == [0, 1, 0, 1]
    assert set(result["sample_weight"]) == {1.0}

    training_input = CardioFeatureEngineering(
        FEATURES, "target_disease", 0.2, 42
    ).build_training_dataframe(dataframe, impute=False)
    assert training_input["bmi"].isna().sum() == 1
    assert training_input["label_stroke"].tolist() == [0, 1, 0, 1]


def test_legacy_target_disease_is_mapped_to_two_labels():
    dataframe = pd.DataFrame(
        [[50, 0, 23.0, 1, 0, 0, 0, 0, 1, 1]],
        columns=FEATURES + ["target_disease"],
    )

    result = CardioFeatureEngineering(FEATURES, "target_disease", 0.2, 42).build_training_dataframe(dataframe)

    assert result.loc[0, "label_heart"] == 1
    assert result.loc[0, "label_stroke"] == 0


def test_evaluator_exposes_imbalance_metrics():
    from src.ml.evaluate import ModelEvaluator

    result = ModelEvaluator().evaluate_binary(
        [0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.4, 0.8, 0.9]
    )

    assert "pr_auc" in result
    assert "f2_score" in result
    assert "specificity" in result
    assert result["recall"] == 1.0
