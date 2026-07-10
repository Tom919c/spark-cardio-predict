class RiskService:
    """Provides basic risk platform service methods."""

    def get_platform_summary(self):
        return {
            "target_groups": ["医疗机构", "健康管理机构", "中老年群体"],
            "core_capabilities": [
                "数据采集与整合",
                "风险预测建模",
                "风险等级评估",
                "干预方案生成",
                "趋势分析与可视化",
            ],
            "status": "foundation_ready",
        }
