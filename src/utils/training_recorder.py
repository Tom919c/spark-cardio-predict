from datetime import datetime
from pathlib import Path


class TrainingResultRecorder:
    """Appends multi-round training summaries to the markdown results log."""

    def __init__(self, output_path="docs/training_results.md"):
        self.output_path = Path(output_path)

    def append_multi_round_result(self, training_result):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.write_text("# 模型训练结果记录\n\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = training_result.get("model_name", "")
        rounds = training_result.get("rounds", 0)
        run_label = training_result.get("run_label", "")
        best_metric = training_result.get("best_metric", "")
        best_result = training_result.get("best_result", {})
        all_round_results = training_result.get("all_round_results", [])

        lines = [
            "\n---\n",
            f"\n## 自动训练记录 {timestamp}\n",
            f"- 模型名称：`{model_name}`\n",
            f"- 训练轮数：`{rounds}`\n",
            f"- 实验标签：`{run_label}`\n",
            f"- 最优选择指标：`{best_metric}`\n",
            "\n### 每轮结果\n",
        ]

        for round_result in all_round_results:
            metrics = round_result.get("metrics", {})
            lines.extend(
                [
                    f"\n#### 第 {round_result.get('round_index')} 轮\n",
                    f"- run_id：`{round_result.get('run_id', '')}`\n",
                    f"- 随机种子：`{round_result.get('round_random_state', '')}`\n",
                    f"- 模型路径：`{round_result.get('model_path', '')}`\n",
                    f"- Accuracy：`{metrics.get('accuracy', '')}`\n",
                    f"- F1 Score：`{metrics.get('f1_score', '')}`\n",
                    f"- ROC AUC：`{metrics.get('roc_auc', '')}`\n",
                ]
            )

        best_metrics = best_result.get("metrics", {})
        lines.extend(
            [
                "\n### 最终最佳结果\n",
                f"- 最佳轮次：`第 {best_result.get('round_index', '')} 轮`\n",
                f"- 最佳模型路径：`{best_result.get('model_path', '')}`\n",
                f"- 最终 Accuracy：`{best_metrics.get('accuracy', '')}`\n",
                f"- 最终 F1 Score：`{best_metrics.get('f1_score', '')}`\n",
                f"- 最终 ROC AUC：`{best_metrics.get('roc_auc', '')}`\n",
            ]
        )

        with self.output_path.open("a", encoding="utf-8") as file:
            file.writelines(lines)
