import pandas as pd


class CardioFeatureEngineering:
    """Preprocessing for the standardized CVD dataset."""

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
        working_frame = self._build_dual_targets(working_frame)

        available_features = [
            column for column in self.feature_columns if column in working_frame.columns
        ]

        model_frame = working_frame[
            available_features + [self.target_column, "heart_risk", "stroke_risk"]
        ].copy()
        model_frame = model_frame.dropna()
        return model_frame

    def _clean_basic_values(self, dataframe):
        cleaned_frame = dataframe.copy()

        numeric_columns = ["age", "bmi", "cholesterol"]
        for column in numeric_columns:
            if column in cleaned_frame.columns:
                cleaned_frame = cleaned_frame[cleaned_frame[column] > 0]

        binary_like_columns = ["gender", "diabetes", "hypertension", "alcohol", "exercise"]
        for column in binary_like_columns:
            if column in cleaned_frame.columns:
                cleaned_frame = cleaned_frame[cleaned_frame[column].isin([0, 1])]

        if "smoker" in cleaned_frame.columns:
            cleaned_frame = cleaned_frame[cleaned_frame["smoker"].isin([0, 1, 2])]

        if self.target_column in cleaned_frame.columns:
            cleaned_frame = cleaned_frame[cleaned_frame[self.target_column].isin([0, 1, 2])]

        if "cholesterol" in cleaned_frame.columns:
            cleaned_frame = cleaned_frame[cleaned_frame["cholesterol"].isin([1, 2, 3])]

        return cleaned_frame

    def _build_dual_targets(self, dataframe):
        target_frame = dataframe.copy()
        if self.target_column in target_frame.columns:
            target_frame["heart_risk"] = (target_frame[self.target_column] == 1).astype(int)
            target_frame["stroke_risk"] = (target_frame[self.target_column] == 2).astype(int)
        return target_frame
