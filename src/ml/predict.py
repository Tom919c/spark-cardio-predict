import joblib
import pandas as pd


class CardioRiskPredictor:
    """Loads trained heart and stroke models and predicts dual risk outputs."""

    def predict(self, model_paths_or_heart_path, stroke_model_path=None, sample=None):
        if isinstance(model_paths_or_heart_path, dict):
            model_paths = model_paths_or_heart_path
            payload = sample if sample is not None else stroke_model_path
        else:
            model_paths = {
                "heart": model_paths_or_heart_path,
                "stroke": stroke_model_path,
            }
            payload = sample

        heart_model = joblib.load(model_paths["heart"])
        stroke_model = joblib.load(model_paths["stroke"])
        sample_frame = self._build_sample_frame(payload)

        heart_label = int(heart_model.predict(sample_frame)[0])
        stroke_label = int(stroke_model.predict(sample_frame)[0])
        heart_probability = self._predict_probability(heart_model, sample_frame)
        stroke_probability = self._predict_probability(stroke_model, sample_frame)

        return {
            "heart_predicted_label": heart_label,
            "stroke_predicted_label": stroke_label,
            "heart_predicted_probability": heart_probability,
            "stroke_predicted_probability": stroke_probability,
        }

    def _predict_probability(self, model, sample_frame):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(sample_frame)[0][1])
        return None

    def _build_sample_frame(self, sample):
        working_sample = dict(sample)
        if "bmi" in working_sample:
            working_sample["bmi"] = round(float(working_sample["bmi"]), 2)
        return pd.DataFrame([working_sample])
