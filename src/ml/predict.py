import joblib
import pandas as pd


class CardioRiskPredictor:
    """Loads a trained model and predicts cardiovascular risk for input samples."""

    def predict(self, model_path, sample):
        model = joblib.load(model_path)
        sample_frame = pd.DataFrame([sample])
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
