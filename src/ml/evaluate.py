from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluates binary subtask models used in the dual-model pipeline."""

    def evaluate_binary(self, y_true, y_pred, y_prob=None):
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_prob is not None:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        return metrics
