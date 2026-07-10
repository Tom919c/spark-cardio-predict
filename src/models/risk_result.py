class RiskResult:
    """Represents a single cardiovascular risk prediction result."""

    def __init__(
        self,
        predicted_label,
        predicted_probability,
        risk_level,
        model_path,
        risk_summary="",
        indicator_insights=None,
        intervention_plan=None,
        key_highlights=None,
    ):
        self.predicted_label = predicted_label
        self.predicted_probability = predicted_probability
        self.risk_level = risk_level
        self.model_path = model_path
        self.risk_summary = risk_summary
        self.indicator_insights = indicator_insights or []
        self.intervention_plan = intervention_plan or {}
        self.key_highlights = key_highlights or []

    def to_dict(self):
        return {
            "predicted_label": self.predicted_label,
            "predicted_probability": self.predicted_probability,
            "risk_level": self.risk_level,
            "model_path": self.model_path,
            "risk_summary": self.risk_summary,
            "indicator_insights": self.indicator_insights,
            "intervention_plan": self.intervention_plan,
            "key_highlights": self.key_highlights,
        }
