import pandas as pd


class CardioFeatureEngineering:
    """First-stage preprocessing for the cardio_train dataset."""

    def __init__(self, feature_columns, target_column, test_size, random_state):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def transform(self, dataframe):
        model_frame = self.build_training_dataframe(dataframe)
        split_index = int(len(model_frame) * (1 - self.test_size))
        train_frame = model_frame.iloc[:split_index].copy()
        test_frame = model_frame.iloc[split_index:].copy()

        return {
            "processed_shape": [
                int(model_frame.shape[0]),
                int(model_frame.shape[1]),
            ],
            "processed_columns": model_frame.columns.tolist(),
            "processed_preview": model_frame.head(5).to_dict(orient="records"),
            "train_shape": [int(train_frame.shape[0]), int(train_frame.shape[1])],
            "test_shape": [int(test_frame.shape[0]), int(test_frame.shape[1])],
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def build_training_dataframe(self, dataframe):
        working_frame = dataframe.copy()
        working_frame = self._clean_basic_values(working_frame)

        if "age" in working_frame.columns:
            # The common cardio dataset stores age in days; convert it for readability.
            working_frame["age_years"] = (working_frame["age"] / 365).round(1)

        if {"height", "weight"}.issubset(working_frame.columns):
            height_in_meters = working_frame["height"] / 100
            working_frame["bmi"] = (
                working_frame["weight"] / (height_in_meters * height_in_meters)
            ).round(2)

        available_features = [
            column for column in self.feature_columns if column in working_frame.columns
        ]
        if "age_years" in working_frame.columns:
            available_features.append("age_years")
        if "bmi" in working_frame.columns:
            available_features.append("bmi")

        model_frame = working_frame[available_features + [self.target_column]].copy()
        model_frame = model_frame.dropna()
        return model_frame

    def _clean_basic_values(self, dataframe):
        cleaned_frame = dataframe.copy()
        cleaned_frame = cleaned_frame.drop_duplicates()

        numeric_columns = [
            "age",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
        ]
        for column in numeric_columns:
            if column in cleaned_frame.columns:
                cleaned_frame = cleaned_frame[cleaned_frame[column] > 0]

        if {"ap_hi", "ap_lo"}.issubset(cleaned_frame.columns):
            cleaned_frame = cleaned_frame[cleaned_frame["ap_hi"] >= cleaned_frame["ap_lo"]]

        return cleaned_frame
