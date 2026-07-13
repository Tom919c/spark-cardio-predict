import joblib
import pandas as pd


class ModelExplainer:
    """Builds single-sample SHAP explanations for trained classifiers."""

    FEATURE_LABELS = {
        "age": "年龄",
        "gender": "性别",
        "bmi": "BMI",
        "cholesterol": "胆固醇情况",
        "diabetes": "糖尿病情况",
        "hypertension": "高血压情况",
        "smoker": "吸烟情况",
        "alcohol": "饮酒情况",
        "exercise": "运动情况",
    }

    FEATURE_VALUE_LABELS = {
        "gender": {0: "女", 1: "男"},
        "cholesterol": {1: "正常", 2: "偏高", 3: "明显升高"},
        "diabetes": {0: "无", 1: "有"},
        "hypertension": {0: "无", 1: "有"},
        "smoker": {0: "从不吸烟", 1: "曾经吸烟", 2: "当前吸烟"},
        "alcohol": {0: "不饮酒", 1: "饮酒"},
        "exercise": {0: "不规律", 1: "规律"},
    }

    def explain_prediction(self, model_path, sample, top_n=3):
        try:
            import shap
        except ImportError:
            return {
                "available": False,
                "method": "shap",
                "message": "SHAP is not installed. Run `pip install shap` to enable explanations.",
                "top_positive_factors": [],
                "top_negative_factors": [],
            }

        model = joblib.load(model_path)
        sample_frame = self._build_sample_frame(sample)

        try:
            explainable_model = getattr(model, "estimator", model)
            if self._is_tree_model(explainable_model):
                explainer = shap.TreeExplainer(explainable_model)
                shap_values = explainer.shap_values(sample_frame)
            else:
                explainer = shap.Explainer(model, sample_frame)
                shap_values = explainer(sample_frame)
        except Exception as exc:
            return {
                "available": False,
                "method": "shap",
                "message": f"SHAP explanation failed: {exc}",
                "top_positive_factors": [],
                "top_negative_factors": [],
            }

        contributions = self._extract_contributions(sample_frame, shap_values)
        ranked = self._rank_contributions(sample_frame.iloc[0].to_dict(), contributions, top_n=top_n)

        return {
            "available": True,
            "method": "shap",
            "message": "SHAP explanation generated successfully.",
            "top_positive_factors": ranked["top_positive_factors"],
            "top_negative_factors": ranked["top_negative_factors"],
        }

    def _build_sample_frame(self, sample):
        working_sample = dict(sample)
        if "bmi" in working_sample:
            working_sample["bmi"] = round(float(working_sample["bmi"]), 2)
        return pd.DataFrame([working_sample])

    def _is_tree_model(self, model):
        model_name = model.__class__.__name__.lower()
        return "forest" in model_name or "tree" in model_name or "boost" in model_name

    def _extract_contributions(self, sample_frame, shap_values):
        if hasattr(shap_values, "values"):
            values = shap_values.values
            if values.ndim == 3:
                return values[0, :, -1]
            return values[0]

        if isinstance(shap_values, list):
            positive_class_values = shap_values[-1]
            return positive_class_values[0]

        if getattr(shap_values, "ndim", 0) == 3:
            return shap_values[0, :, -1]
        return shap_values[0]

    def _rank_contributions(self, sample_dict, contributions, top_n=3):
        contribution_pairs = []
        total_abs = sum(abs(float(value)) for value in contributions) or 1.0

        for feature_name, contribution in zip(sample_dict.keys(), contributions):
            contribution = float(contribution)
            contribution_pairs.append(
                {
                    "feature": feature_name,
                    "feature_label": self.FEATURE_LABELS.get(feature_name, feature_name),
                    "feature_value": sample_dict.get(feature_name),
                    "feature_value_label": self._format_feature_value(
                        feature_name,
                        sample_dict.get(feature_name),
                    ),
                    "contribution": round(contribution, 6),
                    "impact_percent": round(abs(contribution) / total_abs * 100, 2),
                    "direction": "increase" if contribution >= 0 else "decrease",
                }
            )

        positive = sorted(
            [item for item in contribution_pairs if item["contribution"] > 0],
            key=lambda item: item["contribution"],
            reverse=True,
        )[:top_n]
        negative = sorted(
            [item for item in contribution_pairs if item["contribution"] < 0],
            key=lambda item: item["contribution"],
        )[:top_n]

        return {
            "top_positive_factors": positive,
            "top_negative_factors": negative,
        }

    def _format_feature_value(self, feature_name, feature_value):
        value_map = self.FEATURE_VALUE_LABELS.get(feature_name)
        if value_map is not None:
            try:
                normalized_value = int(feature_value)
            except (TypeError, ValueError):
                normalized_value = feature_value
            return value_map.get(normalized_value, str(feature_value))

        if feature_name == "bmi":
            try:
                return f"{float(feature_value):.1f}"
            except (TypeError, ValueError):
                return str(feature_value)

        return str(feature_value)
