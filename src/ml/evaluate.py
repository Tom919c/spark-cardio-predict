from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluates binary subtask models used in the dual-model pipeline."""

    def evaluate_binary(self, y_true, y_pred, y_prob=None):
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        specificity = true_negative / max(true_negative + false_positive, 1)
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score": round(float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)), 4),
            "f2_score": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "specificity": round(float(specificity), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "positive_rate": round(float(sum(y_true) / max(len(y_true), 1)), 6),
            "confusion_matrix": matrix.tolist(),
        }
        if y_prob is not None:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            metrics["brier_score"] = round(float(brier_score_loss(y_true, y_prob)), 4)
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        return metrics
