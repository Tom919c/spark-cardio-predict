def build_risk_interpretation(sample, predicted_probability, risk_level):
    """Builds a readable explanation of why the sample is low/medium/high risk."""

    highlights = []
    if sample.get("ap_hi", 0) >= 140 or sample.get("ap_lo", 0) >= 90:
        highlights.append("血压偏高，提示心脑血管负担增加")
    if sample.get("cholesterol", 1) >= 2:
        highlights.append("胆固醇水平偏高，存在动脉粥样硬化风险")
    if sample.get("gluc", 1) >= 2:
        highlights.append("血糖水平偏高，需要关注代谢风险")
    if sample.get("smoke", 0) == 1:
        highlights.append("存在吸烟行为，会增加心脑血管事件概率")
    if sample.get("alco", 0) == 1:
        highlights.append("存在饮酒行为，建议关注长期心血管影响")
    if sample.get("active", 1) == 0:
        highlights.append("缺乏规律运动，不利于心血管健康管理")

    if not highlights:
        highlights.append("当前主要指标整体相对平稳，未见明显高危特征")

    summary = {
        "low": "当前模型评估为低风险，建议持续保持健康生活方式。",
        "medium": "当前模型评估为中风险，建议尽快加强生活方式干预并定期复查。",
        "high": "当前模型评估为高风险，建议尽快前往医疗机构进一步检查与评估。",
        "unknown": "当前风险等级暂无法明确，建议结合更多临床信息判断。",
    }

    return {
        "risk_summary": summary.get(risk_level, summary["unknown"]),
        "risk_probability_percent": round(predicted_probability * 100, 2)
        if predicted_probability is not None
        else None,
        "key_highlights": highlights,
    }


def build_indicator_insights(sample):
    """Generates indicator-level interpretations for major health features."""

    insights = []

    systolic = sample.get("ap_hi")
    diastolic = sample.get("ap_lo")
    if systolic is not None and diastolic is not None:
        blood_pressure_status = "正常"
        if systolic >= 140 or diastolic >= 90:
            blood_pressure_status = "偏高"
        insights.append(
            {
                "indicator": "血压",
                "value": f"{systolic}/{diastolic}",
                "status": blood_pressure_status,
                "comment": "血压偏高时建议结合家庭血压监测与医疗随访。",
            }
        )

    cholesterol = sample.get("cholesterol")
    if cholesterol is not None:
        status = "正常" if cholesterol == 1 else "偏高"
        insights.append(
            {
                "indicator": "胆固醇",
                "value": cholesterol,
                "status": status,
                "comment": "胆固醇升高与动脉粥样硬化风险相关。",
            }
        )

    gluc = sample.get("gluc")
    if gluc is not None:
        status = "正常" if gluc == 1 else "偏高"
        insights.append(
            {
                "indicator": "血糖",
                "value": gluc,
                "status": status,
                "comment": "血糖异常时建议关注糖代谢与心血管共病风险。",
            }
        )

    bmi = sample.get("bmi")
    if bmi is not None:
        status = "正常"
        if bmi >= 24:
            status = "超重或肥胖倾向"
        insights.append(
            {
                "indicator": "BMI",
                "value": bmi,
                "status": status,
                "comment": "BMI 偏高会增加高血压、糖脂代谢异常风险。",
            }
        )

    return insights
