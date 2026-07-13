class InterventionPlan:
    """Represents a personalized intervention plan."""

    def __init__(self, risk_level, suggestions):
        self.risk_level = risk_level
        self.suggestions = suggestions

    def to_dict(self):
        return {
            "risk_level": self.risk_level,
            "suggestions": self.suggestions,
        }
