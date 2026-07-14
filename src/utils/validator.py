class DatasetValidator:
    """Validates dataset structure before model preprocessing."""

    def __init__(self, required_columns):
        self.required_columns = required_columns

    def validate_columns(self, actual_columns):
        missing_columns = [
            column for column in self.required_columns if column not in actual_columns
        ]
        return {
            "valid": len(missing_columns) == 0,
            "required_columns": self.required_columns,
            "missing_columns": missing_columns,
            "actual_columns": actual_columns,
        }


import math


class PredictionInputValidator:
    """Validates required fields for single-person risk prediction."""

    NUMERIC_RANGES = {
        "age": (18, 95),
        "bmi": (10.3, 79.8),
        "gender": (0, 1),
        "cholesterol": (1, 3),
        "diabetes": (0, 1),
        "hypertension": (0, 1),
        "smoker": (0, 2),
        "alcohol": (0, 1),
        "exercise": (0, 1),
    }
    OPTIONAL_NUMERIC_RANGES = {
        "systolic_bp": (60, 260),
        "diastolic_bp": (30, 180),
        "fasting_glucose": (1, 40),
        "family_history": (0, 1),
    }

    def __init__(self, required_fields):
        self.required_fields = required_fields

    def validate_payload(self, payload):
        missing_fields = [
            field for field in self.required_fields if field not in payload
        ]
        invalid_fields = {}
        for field in self.required_fields:
            if field in missing_fields:
                continue
            try:
                value = float(payload[field])
            except (TypeError, ValueError):
                invalid_fields[field] = "必须是数字"
                continue
            if not math.isfinite(value):
                invalid_fields[field] = "必须是有限数字"
                continue
            limits = self.NUMERIC_RANGES.get(field)
            if limits and not limits[0] <= value <= limits[1]:
                invalid_fields[field] = f"取值范围为 {limits[0]}~{limits[1]}"
                continue
            if field != "bmi" and value != int(value):
                invalid_fields[field] = "必须是整数编码"
        for field, limits in self.OPTIONAL_NUMERIC_RANGES.items():
            if field not in payload or payload[field] in (None, ""):
                continue
            try:
                value = float(payload[field])
            except (TypeError, ValueError):
                invalid_fields[field] = "必须是数字"
                continue
            if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
                invalid_fields[field] = f"取值范围为 {limits[0]}~{limits[1]}"
                continue
            if field == "family_history" and value != int(value):
                invalid_fields[field] = "必须是整数编码"
        return {
            "valid": len(missing_fields) == 0 and not invalid_fields,
            "required_fields": self.required_fields,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
        }
