import joblib
import pandas as pd


class CardioRiskPredictor:
    """Loads a trained model and predicts cardiovascular risk for input samples."""

    def predict(self, model_path, sample):
        model = joblib.load(model_path)
        sample_frame = self._build_sample_frame(sample)
        predicted_label = int(model.predict(sample_frame)[0])
        probability = self._predict_probability(model, sample_frame)
        return {
            "predicted_label": predicted_label,
            "predicted_probability": probability,
        }

    def _predict_probability(self, model, sample_frame):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(sample_frame)[0][1])
        return None

    def _build_sample_frame(self, sample):
        working_sample = dict(sample)
        if "age" in working_sample and "age_years" not in working_sample:
            working_sample["age_years"] = round(working_sample["age"] / 365, 1)
        if (
            "height" in working_sample
            and "weight" in working_sample
            and "bmi" not in working_sample
        ):
            height_in_meters = working_sample["height"] / 100
            working_sample["bmi"] = round(
                working_sample["weight"] / (height_in_meters * height_in_meters), 2
            )
        return pd.DataFrame([working_sample])
