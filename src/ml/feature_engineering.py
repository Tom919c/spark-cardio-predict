"""特征工程：从标准化数据构建无数据泄漏的双标签训练集。"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class CardioFeatureEngineering:
    """构建无泄漏的双标签训练数据集。"""

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
            "processed_shape": [int(model_frame.shape[0]), int(model_frame.shape[1])],
            "processed_columns": model_frame.columns.tolist(),
            "processed_preview": model_frame.head(5).to_dict(orient="records"),
            "train_shape": [int(train_frame.shape[0]), int(train_frame.shape[1])],
            "test_shape": [int(test_frame.shape[0]), int(test_frame.shape[1])],
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def build_training_dataframe(self, dataframe):
        missing = [c for c in self.feature_columns if c not in dataframe]
        if missing:
            raise ValueError(f"数据缺少必要特征列: {missing}")

        working_frame = self._build_dual_targets(dataframe.copy())
        feature_frame = working_frame[self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
        imputed = pd.DataFrame(
            imputer.fit_transform(feature_frame),
            columns=self.feature_columns,
            index=working_frame.index,
        )
        model_frame = self._normalise_feature_values(imputed)
        model_frame["label_heart"] = working_frame["label_heart"].astype(int)
        model_frame["label_stroke"] = working_frame["label_stroke"].astype(int)
        model_frame["sample_weight"] = self._build_sample_weight(working_frame)
        return model_frame

    def _build_dual_targets(self, dataframe):
        target_frame = dataframe.copy()
        # 新版 DWD 优先使用独立标签；旧三分类数据只作为兼容输入。
        if "label_heart" not in target_frame:
            if self.target_column not in target_frame:
                raise ValueError("数据缺少 label_heart 和兼容目标列 target_disease。")
            target_frame["label_heart"] = (target_frame[self.target_column] == 1).astype(int)
        if "label_stroke" not in target_frame:
            if self.target_column not in target_frame:
                raise ValueError("数据缺少 label_stroke 和兼容目标列 target_disease。")
            target_frame["label_stroke"] = (target_frame[self.target_column] == 2).astype(int)
        return target_frame

    def _normalise_feature_values(self, dataframe):
        frame = dataframe.copy()
        # 插补输出为连续值，离散医疗编码必须回写到模型约定的合法取值范围。
        frame["age"] = frame["age"].clip(18, 95).round().astype(int)
        frame["bmi"] = frame["bmi"].clip(10.3, 79.8).round(2)
        frame["cholesterol"] = frame["cholesterol"].clip(1, 3).round().astype(int)
        frame["smoker"] = frame["smoker"].clip(0, 2).round().astype(int)
        for col in ("gender", "diabetes", "hypertension", "alcohol", "exercise"):
            frame[col] = frame[col].clip(0, 1).round().astype(int)
        return frame

    @staticmethod
    def _build_sample_weight(dataframe):
        if "region" not in dataframe:
            return pd.Series(1.0, index=dataframe.index)
        # 本地人群在混合训练中提高权重，未标记地域的数据保持默认权重。
        local = dataframe["region"].astype(str).str.contains("成都|四川|中国", regex=True)
        return local.map({True: 4.0, False: 1.0}).fillna(1.0)
