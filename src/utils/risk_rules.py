def build_risk_interpretation(
    sample, heart_probability, stroke_probability, final_category, risk_level=None
):
    """Builds readable explanations for the dual-model output."""

    highlights = []
    if sample.get("hypertension", 0) == 1:
        highlights.append("存在高血压特征，提示心脑血管负担增加")
    if sample.get("cholesterol", 1) >= 2:
        highlights.append("胆固醇水平偏高，存在动脉粥样硬化风险")
    if sample.get("diabetes", 0) == 1:
        highlights.append("存在糖尿病特征，需要关注代谢与血管风险")
    if sample.get("smoker", 0) >= 1:
        highlights.append("存在吸烟相关行为，会增加心脏和脑卒中风险")
    if sample.get("alcohol", 0) == 1:
        highlights.append("存在饮酒行为，建议关注长期心血管影响")
    if sample.get("exercise", 1) == 0:
        highlights.append("缺乏规律运动，不利于心脑血管健康管理")
    if sample.get("bmi", 0) >= 24:
        highlights.append("BMI 偏高，提示超重或肥胖相关风险")

    if not highlights:
        highlights.append("当前主要指标整体相对平稳，未见明显高危特征")

    level_code = risk_level.get("code", 1) if isinstance(risk_level, dict) else 1
    summary = {
        0: (
            "当前两项独立模型均未达到组合高风险阈值，综合风险处于关注范围，"
            "建议继续管理危险因素。"
            if level_code > 1
            else "当前评估为健康状态，未发现明显心脏或脑卒中高风险信号。"
        ),
        1: "当前评估更偏向心脏负面事件风险，需要重点关注心脏相关危险因素。",
        2: "当前评估更偏向脑卒中风险，需要重点关注脑血管相关危险因素。",
        3: "当前评估提示心脏和脑卒中双重风险，建议尽快进行系统性检查与干预。",
    }

    return {
        "risk_summary": summary.get(final_category, summary[0]),
        "heart_probability_percent": round(heart_probability * 100, 2)
        if heart_probability is not None
        else None,
        "stroke_probability_percent": round(stroke_probability * 100, 2)
        if stroke_probability is not None
        else None,
        "key_highlights": highlights,
    }


def build_indicator_insights(sample):
    """Generates indicator-level explanations for the standardized CVD dataset."""

    insights = []

    hypertension = sample.get("hypertension")
    if hypertension is not None:
        status = "正常" if hypertension == 0 else "偏高风险"
        insights.append(
            {
                "indicator": "高血压",
                "value": hypertension,
                "status": status,
                "comment": "高血压是心脑血管疾病的重要危险因素。",
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

    diabetes = sample.get("diabetes")
    if diabetes is not None:
        status = "正常" if diabetes == 0 else "存在风险"
        insights.append(
            {
                "indicator": "糖尿病",
                "value": diabetes,
                "status": status,
                "comment": "糖尿病与心脑血管共病风险密切相关。",
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

    smoker = sample.get("smoker")
    if smoker is not None:
        status_map = {0: "从不吸烟", 1: "曾经吸烟", 2: "当前吸烟"}
        insights.append(
            {
                "indicator": "吸烟状态",
                "value": smoker,
                "status": status_map.get(smoker, "未知"),
                "comment": "吸烟与心脏负面事件和脑卒中风险均相关。",
            }
        )

    return insights
