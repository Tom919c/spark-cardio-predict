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
