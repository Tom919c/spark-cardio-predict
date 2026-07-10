from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


class ModelEvaluator:
    """Evaluates trained model performance using baseline classification metrics."""

    def evaluate(self, y_true, y_pred, y_prob=None):
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred)), 4),
            "classification_report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }
        if y_prob is not None:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        return metrics
