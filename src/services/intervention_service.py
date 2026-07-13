class InterventionService:
    """Builds practical suggestions from risk factors and a five-level assessment."""

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
            1: "建议保持现有健康管理习惯，每 12 个月复查一次关键指标。",
            2: "建议低盐低脂饮食，每 6 个月复查血压、血脂等指标。",
            3: "建议建立个人健康档案，并每 3 个月完成一次随访复查。",
            4: "建议尽快在医生指导下进行血压、血脂和心脑血管专项评估，每月复查。",
            5: "建议 2 周内前往医疗机构进行全面临床检查并接受专业指导。",
        }
        level_code = risk_level.get("code", 1) if isinstance(risk_level, dict) else risk_level
        suggestions.append(level_guidance.get(level_code, level_guidance[1]))

        return {
            "risk_level": risk_level,
            "suggestions": suggestions,
        }
