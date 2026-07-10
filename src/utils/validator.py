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


class PredictionInputValidator:
    """Validates required fields for single-person risk prediction."""

    def __init__(self, required_fields):
        self.required_fields = required_fields

    def validate_payload(self, payload):
        missing_fields = [
            field for field in self.required_fields if field not in payload
        ]
        return {
            "valid": len(missing_fields) == 0,
            "required_fields": self.required_fields,
            "missing_fields": missing_fields,
        }
