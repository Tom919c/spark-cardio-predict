from datetime import datetime
from pathlib import Path


class TrainingResultRecorder:
    """Appends training summaries to the markdown results log."""

    def __init__(self, output_path="docs/training_results.md"):
        self.output_path = Path(output_path)

    def append_multi_round_result(self, training_result):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.write_text("# Training Results\n\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "\n---\n",
            f"\n## Auto Training Record {timestamp}\n",
            f"- Strategy: `{training_result.get('strategy', '')}`\n",
        ]

        if "targets" in training_result:
            for target_name, target_result in training_result["targets"].items():
                metrics = target_result.get("metrics", {})
                lines.extend(
                    [
                        f"\n### {target_name}\n",
                        f"- Run ID: `{target_result.get('run_id', '')}`\n",
                        f"- Model Path: `{target_result.get('model_path', '')}`\n",
                        f"- Accuracy: `{metrics.get('accuracy', '')}`\n",
                        f"- F1 Score: `{metrics.get('f1_score', '')}`\n",
                        f"- ROC AUC: `{metrics.get('roc_auc', '')}`\n",
                    ]
                )
            registry = training_result.get("registry", {})
            if registry:
                lines.append(f"- Manifest Updated At: `{registry.get('updated_at', '')}`\n")
        else:
            model_name = training_result.get("model_name", "")
            rounds = training_result.get("rounds", 0)
            run_label = training_result.get("run_label", "")
            best_metric = training_result.get("best_metric", "")
            best_result = training_result.get("best_result", {})
            all_round_results = training_result.get("all_round_results", [])
            lines.extend(
                [
                    f"- Model Name: `{model_name}`\n",
                    f"- Rounds: `{rounds}`\n",
                    f"- Run Label: `{run_label}`\n",
                    f"- Best Metric: `{best_metric}`\n",
                ]
            )
            for round_result in all_round_results:
                metrics = round_result.get("metrics", {})
                lines.extend(
                    [
                        f"\n### Round {round_result.get('round_index', '')}\n",
                        f"- Run ID: `{round_result.get('run_id', '')}`\n",
                        f"- Model Path: `{round_result.get('model_path', '')}`\n",
                        f"- Accuracy: `{metrics.get('accuracy', '')}`\n",
                        f"- F1 Score: `{metrics.get('f1_score', '')}`\n",
                        f"- ROC AUC: `{metrics.get('roc_auc', '')}`\n",
                    ]
                )
            if best_result:
                lines.extend(
                    [
                        "\n### Best Result\n",
                        f"- Best Round: `{best_result.get('round_index', '')}`\n",
                        f"- Best Model Path: `{best_result.get('model_path', '')}`\n",
                    ]
                )

        with self.output_path.open("a", encoding="utf-8") as file:
            file.writelines(lines)
