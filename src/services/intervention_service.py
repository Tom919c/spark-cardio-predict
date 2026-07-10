class InterventionService:
    """Builds personalized intervention suggestions based on risk result and sample."""

    def build_plan(self, sample, risk_level):
        suggestions = []

        if sample.get("ap_hi", 0) >= 140 or sample.get("ap_lo", 0) >= 90:
            suggestions.append("建议连续监测血压，并根据医生建议进行血压管理。")
        if sample.get("cholesterol", 1) >= 2:
            suggestions.append("建议减少高脂饮食摄入，增加膳食纤维与优质蛋白。")
        if sample.get("gluc", 1) >= 2:
            suggestions.append("建议控制精制糖摄入，并关注空腹血糖与糖化血红蛋白。")
        if sample.get("smoke", 0) == 1:
            suggestions.append("建议尽快戒烟，减少烟草对心脑血管系统的长期损害。")
        if sample.get("alco", 0) == 1:
            suggestions.append("建议减少饮酒频率和饮酒量，避免长期饮酒相关风险。")
        if sample.get("active", 1) == 0:
            suggestions.append("建议每周保持规律有氧运动，逐步提高身体活动水平。")

        level_guidance = {
            "low": "建议保持现有健康管理习惯，每 6 到 12 个月复查一次关键指标。",
            "medium": "建议在 3 到 6 个月内复查血压、血糖、血脂，并加强生活方式干预。",
            "high": "建议尽快前往医疗机构做进一步检查，必要时进行系统化干预管理。",
            "unknown": "建议结合更多临床与检查信息后再制定更完整干预方案。",
        }
        suggestions.append(level_guidance.get(risk_level, level_guidance["unknown"]))

        return {
            "risk_level": risk_level,
            "suggestions": suggestions,
        }
