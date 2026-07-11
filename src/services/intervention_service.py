class InterventionService:
    """Builds intervention suggestions based on the combined dual-model category."""

    def build_plan(self, sample, risk_level):
        suggestions = []

        if sample.get("hypertension", 0) == 1:
            suggestions.append("建议持续监测血压，并根据医生建议进行血压管理。")
        if sample.get("cholesterol", 1) >= 2:
            suggestions.append("建议减少高脂饮食摄入，增加膳食纤维与优质蛋白。")
        if sample.get("diabetes", 0) == 1:
            suggestions.append("建议控制精制糖摄入，并规律复查糖代谢相关指标。")
        if sample.get("smoker", 0) >= 1:
            suggestions.append("建议尽快戒烟或持续戒烟，降低心脑血管风险。")
        if sample.get("alcohol", 0) == 1:
            suggestions.append("建议减少饮酒频率和饮酒量，避免长期饮酒相关风险。")
        if sample.get("exercise", 1) == 0:
            suggestions.append("建议每周保持规律有氧运动，逐步提高身体活动水平。")
        if sample.get("bmi", 0) >= 24:
            suggestions.append("建议控制体重，优先通过饮食调整和规律运动降低 BMI。")

        level_guidance = {
            0: "建议保持现有健康管理习惯，每 6 到 12 个月复查一次关键指标。",
            1: "建议优先关注心脏相关检查，如血压、血脂、心电图或心脏专项评估。",
            2: "建议优先关注脑卒中相关风险管理，如血压控制、血管评估和神经系统随访。",
            3: "建议尽快前往医疗机构做系统化检查，同时关注心脏和脑血管双重风险管理。",
        }
        suggestions.append(level_guidance.get(risk_level, level_guidance[0]))

        return {
            "risk_level": risk_level,
            "suggestions": suggestions,
        }
