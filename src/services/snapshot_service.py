"""多批次机构数据快照比较服务。"""

from __future__ import annotations


class SnapshotService:
    """只比较同一统计口径下的历史快照，不进行时序预测。"""

    REQUIRED_COMPATIBILITY = ("coverage_level", "coverage_name", "model_version")

    def compare(self, snapshots):
        valid = [item for item in snapshots if item.get("data_period")]
        valid.sort(key=lambda item: str(item["data_period"]))
        if not valid:
            return {"available": False, "message": "没有包含 data_period 的历史快照。", "series": []}

        compatibility = self._compatibility(valid)
        if not compatibility["compatible"]:
            return {
                "available": False,
                "message": compatibility["message"],
                "series": [],
            }
        return {
            "available": True,
            "message": "快照比较结果生成成功。",
            "series": [
                {
                    "data_period": item["data_period"],
                    "coverage_name": item.get("coverage_name"),
                    "total_residents": item.get("total_residents"),
                    "heart_risk_rate": item.get("heart_risk_rate"),
                    "stroke_risk_rate": item.get("stroke_risk_rate"),
                    "comorbidity_rate": item.get("comorbidity_rate"),
                    "risk_metric_source": item.get("risk_metric_source"),
                    "model_version": item.get("model_version", "unknown"),
                }
                for item in valid
            ],
            "interpretation": "快照反映不同数据批次的统计变化，不代表干预的因果效果。",
        }

    def _compatibility(self, snapshots):
        first = snapshots[0]
        for current in snapshots[1:]:
            for field in self.REQUIRED_COMPATIBILITY:
                if current.get(field) != first.get(field):
                    return {
                        "compatible": False,
                        "message": f"快照的 {field} 不一致，不能放入同一条趋势线。",
                    }
        return {"compatible": True, "message": "快照口径一致。"}
