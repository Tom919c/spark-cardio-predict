"""Build the versioned local intervention knowledge base from Markdown extracts.

The source files are curated extracts, not machine-readable clinical guidelines.
This builder preserves the original wording and adds conservative automation
metadata. It intentionally keeps medication and acute-care rules available for
review while preventing them from becoming automatic patient advice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "知识库构建材料文档"
OUTPUT_PATH = ROOT / "resources" / "knowledge" / "intervention_rules.json"

RULE_RE = re.compile(
    r"### \[规则编号：(?P<rule_id>[^\]]+)\](?P<body>.*?)(?=^### \[规则编号：|\Z)",
    re.MULTILINE | re.DOTALL,
)
FIELD_RE = re.compile(
    r"^\*\s+\*\*(?P<name>[^*]+)\*\*:\s*(?P<value>.*?)(?=^\*\s+\*\*|^##|^---|\Z)",
    re.MULTILINE | re.DOTALL,
)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
ROMAN_LEVELS = {"Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5}

SOURCE_META = {
    "2024脑卒中防治报告解读_核心干预规则.md": {
        "namespace": "stroke-report-2024",
        "domain": "stroke",
        "source": "《2024年中国脑卒中防治报告》解读及其引用的《脑血管病防治指南（2024年版）》",
        "evidence_level": "interpretive_review",
        "uncertainty_note": "本材料为解读性综述，不是正式指南全文；规则主要用于来源展示和人工复核。",
        "scope": ["stroke"],
    },
    "脑卒中防治核心干预规则提取说明书.md": {
        "namespace": "stroke-primary-2021",
        "domain": "stroke",
        "source": "《中国脑卒中防治指导规范（2021年版）》第一章",
        "evidence_level": "guideline_derived",
        "uncertainty_note": "本文件是从指南章节提取的规则集，不替代完整指南、临床评估或医生处方。",
        "scope": ["stroke"],
    },
    "心血管病一级预防核心干预规则提取说明书.md": {
        "namespace": "cvd-primary-2020",
        "domain": "cardiovascular",
        "source": "《中国心血管病一级预防指南（2020版）》",
        "evidence_level": "guideline_derived",
        "uncertainty_note": "本文件是从指南提取的规则集，适用于一级预防背景，不替代个体化临床决策。",
        "scope": ["heart", "stroke"],
    },
    "中国居民膳食指南2022_核心干预规则.md": {
        "namespace": "diet-2022",
        "domain": "nutrition",
        "source": "《中国居民膳食指南（2022）》核心信息",
        "evidence_level": "guideline_derived",
        "uncertainty_note": "多数定量建议以一般成年人及1600～2400kcal能量需要量为参考，特殊人群需单独评估。",
        "scope": ["heart", "stroke"],
    },
}

PROJECT_RULES = [
    {
        "rule_id": "hypertension-monitoring",
        "source_rule_id": "PROJECT_001",
        "factor": "hypertension",
        "auto_factors": ["hypertension"],
        "risk_levels": [1, 2, 3, 4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "建议定期监测血压，并按照医生建议进行血压管理。",
        "boundary_notes": "不替代医生诊断和处方。",
        "priority": 20,
    },
    {
        "rule_id": "cholesterol-diet",
        "source_rule_id": "PROJECT_002",
        "factor": "cholesterol",
        "auto_factors": ["cholesterol"],
        "risk_levels": [1, 2, 3, 4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "建议减少高脂、高盐饮食，增加蔬菜、水果和膳食纤维摄入。",
        "boundary_notes": "具体饮食方案应结合医生或营养师建议。",
        "priority": 20,
    },
    {
        "rule_id": "diabetes-management",
        "source_rule_id": "PROJECT_003",
        "factor": "diabetes",
        "auto_factors": ["diabetes"],
        "risk_levels": [1, 2, 3, 4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "建议控制精制糖摄入，并按医嘱复查血糖和糖化血红蛋白。",
        "boundary_notes": "不得根据本系统结果自行调整降糖药物。",
        "priority": 20,
    },
    {
        "rule_id": "smoking-cessation",
        "source_rule_id": "PROJECT_004",
        "factor": "smoker",
        "auto_factors": ["smoker"],
        "risk_levels": [1, 2, 3, 4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "建议尽快戒烟并减少二手烟暴露，必要时寻求专业戒烟支持。",
        "boundary_notes": "戒烟过程中的不适应咨询专业人员。",
        "priority": 20,
    },
    {
        "rule_id": "activity-and-weight",
        "source_rule_id": "PROJECT_005",
        "factor": "exercise",
        "auto_factors": ["exercise"],
        "risk_levels": [1, 2, 3, 4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "建议在身体条件允许时逐步增加规律身体活动，并结合饮食管理控制体重。",
        "boundary_notes": "运动强度应根据个人健康状况和医生建议调整。",
        "priority": 20,
    },
    {
        "rule_id": "high-risk-follow-up",
        "source_rule_id": "PROJECT_006",
        "factor": "risk_level",
        "auto_factors": ["risk_level"],
        "risk_levels": [4, 5],
        "disease_scope": ["heart", "stroke"],
        "advice": "模型评估结果提示需要重点关注，建议尽快到医疗机构进行专业评估。",
        "boundary_notes": "本条建议不构成急诊判断；出现急性症状时应立即寻求急救。",
        "priority": 100,
    },
]

UNSAFE_TERMS = (
    "药物",
    "服用",
    "阿司匹林",
    "抗凝",
    "他汀",
    "降压药",
    "降糖药",
    "胰岛素",
    "二甲双胍",
    "SGLT",
    "GLP-1",
    "PCSK9",
    "ACEI",
    "ARB",
    "CCB",
    "利尿剂",
    "溶栓",
    "取栓",
    "急性",
    "入院",
    "机械",
    "剂量",
)


def _levels(value: str) -> list[int]:
    values = [ROMAN_LEVELS[item] for item in re.findall("[ⅠⅡⅢⅣⅤ]", value)]
    if len(values) >= 2 and any(marker in value for marker in ("～", "至", "-", "—")):
        return list(range(min(values), max(values) + 1))
    return sorted(set(values))


def _fields(body: str) -> dict[str, str]:
    fields = {}
    for match in FIELD_RE.finditer(body):
        value = " ".join(match.group("value").split())
        fields[match.group("name").strip()] = value
    return fields


def _section(text: str, start: int) -> str:
    current = "未分类"
    for match in SECTION_RE.finditer(text, 0, start):
        current = match.group(1).strip()
    return current


def _factors(trigger: str, advice: str) -> list[str]:
    text = trigger + advice
    candidates = []
    terms = (
        ("hypertension", ("高血压", "血压")),
        ("diabetes", ("糖尿病", "血糖", "糖耐量")),
        ("cholesterol", ("胆固醇", "血脂", "LDL", "HDL", "甘油三", "TG")),
        ("smoker", ("吸烟", "戒烟")),
        ("alcohol", ("饮酒", "酒精", "酒")),
        ("exercise", ("运动", "身体活动", "锻炼", "体力活动")),
        ("exercise", ("体重", "BMI", "肥胖", "超重", "减重")),
        ("age", ("年龄", "岁")),
        ("sleep", ("睡眠", "作息")),
    )
    for factor, keywords in terms:
        if any(keyword in text for keyword in keywords) and factor not in candidates:
            candidates.append(factor)
    return candidates


def _rule_class(trigger: str, advice: str) -> str:
    text = trigger + advice
    if any(term in text for term in ("溶栓", "取栓", "急性", "入院", "发病", "机械")):
        return "acute_treatment"
    if any(term in text for term in UNSAFE_TERMS):
        return "medication_or_clinical_decision"
    if any(term in text for term in ("筛查", "监测", "测量", "复查", "检测")):
        return "screening_or_monitoring"
    if any(
        term in text
        for term in (
            "饮食",
            "蔬菜",
            "水果",
            "谷",
            "盐",
            "钠",
            "运动",
            "身体活动",
            "体重",
            "减重",
            "戒烟",
            "吸烟",
            "饮酒",
            "酒",
            "睡眠",
            "作息",
            "压力",
            "情绪",
            "饮水",
            "烹调油",
            "脂肪",
        )
    ):
        return "lifestyle"
    if "风险" in text or "分层" in text or "评估" in text:
        return "risk_definition"
    return "evidence_or_context"


def _automation(rule_class: str, factors: list[str], trigger: str, advice: str, meta: dict) -> bool:
    if meta["namespace"] == "stroke-report-2024":
        return False
    if not factors or any(factor in {"age", "sleep"} for factor in factors):
        return False
    if rule_class not in {"lifestyle", "screening_or_monitoring"}:
        return False
    text = trigger + advice
    return not any(term in text for term in UNSAFE_TERMS)


def parse_document(path: Path) -> list[dict]:
    meta = SOURCE_META[path.name]
    text = path.read_text(encoding="utf-8")
    rules = []
    for match in RULE_RE.finditer(text):
        source_rule_id = match.group("rule_id").strip()
        fields = _fields(match.group("body"))
        trigger = fields.get("适用危险因素 (Trigger Factors)", "")
        risk_levels = _levels(fields.get("适用风险等级 (Target Risk Levels)", ""))
        advice = fields.get("干预建议原文 (Intervention Advice)", "")
        boundary = fields.get("禁忌与注意事项 (Contraindications)", "")
        rule_class = _rule_class(trigger, advice)
        factors = _factors(trigger, advice)
        automation_safe = _automation(rule_class, factors, trigger, advice, meta)
        rule_id = f"{meta['namespace']}-{source_rule_id.lower().replace('_', '-') }"
        rules.append(
            {
                "rule_id": rule_id,
                "source_rule_id": source_rule_id,
                "domain": meta["domain"],
                "rule_class": rule_class,
                "factor": factors[0] if factors else "manual_review",
                "auto_factors": factors if automation_safe else [],
                "risk_levels": risk_levels,
                "disease_scope": meta["scope"],
                "target_population": trigger,
                "trigger_condition": trigger,
                "advice": advice,
                "boundary_notes": boundary,
                "contraindication": boundary,
                "source": meta["source"],
                "source_doc": path.name,
                "source_section": _section(text, match.start()),
                "source_excerpt": f"{advice} 注意事项：{boundary}",
                "evidence_level": meta["evidence_level"],
                "uncertainty_note": meta["uncertainty_note"],
                "automation_safe": automation_safe,
                "review_mode": "auto" if automation_safe else "manual",
                "priority": 30 if automation_safe else 0,
                "version": meta["namespace"],
                "review_date": "2026-07-13",
            }
        )
    return rules


def build() -> dict:
    rules = []
    for project_rule in PROJECT_RULES:
        rules.append(
            {
                **project_rule,
                "domain": "project_policy",
                "rule_class": "lifestyle" if project_rule["factor"] != "risk_level" else "follow_up",
                "target_population": "项目当前双模型风险评估结果与可识别危险因素",
                "trigger_condition": project_rule["factor"],
                "contraindication": project_rule["boundary_notes"],
                "source": "项目审核版基础健康管理规则",
                "source_doc": "项目审核版基础健康管理规则",
                "source_section": "项目安全默认规则",
                "source_excerpt": f"{project_rule['advice']} 注意事项：{project_rule['boundary_notes']}",
                "evidence_level": "project_reviewed_template",
                "uncertainty_note": "项目固定健康管理模板，不构成诊断、处方或急救判断。",
                "automation_safe": True,
                "review_mode": "auto",
                "priority": project_rule["priority"],
                "version": "project-policy-1.0",
                "review_date": "2026-07-13",
            }
        )
    for path in sorted(SOURCE_DIR.glob("*.md")):
        if path.name not in SOURCE_META:
            continue
        rules.extend(parse_document(path))
    rule_ids = [rule["rule_id"] for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("知识库规则 ID 冲突，请检查来源命名空间。")
    return {
        "version": "phase2-2.0",
        "updated_at": "2026-07-13",
        "description": "来自项目材料的可审核心脑血管健康干预规则；自动建议仅使用 automation_safe=true 的规则。",
        "sources": [
            {
                "source_doc": filename,
                "namespace": metadata["namespace"],
                "domain": metadata["domain"],
                "source": metadata["source"],
                "evidence_level": metadata["evidence_level"],
                "uncertainty_note": metadata["uncertainty_note"],
            }
            for filename, metadata in SOURCE_META.items()
        ],
        "automation_policy": {
            "automatic_rules_only": True,
            "manual_review_required_for": [
                "药物、剂量、抗凝、阿司匹林、降压/降脂/降糖药",
                "急性卒中溶栓、取栓和入院处置",
                "需要实验室、影像、病史或禁忌证综合判断的规则",
            ],
            "disclaimer": "系统建议用于健康管理提示，不构成诊断、处方或急救判断。",
        },
        "rules": rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    content = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    safe = sum(rule["automation_safe"] for rule in content["rules"])
    print(f"built {len(content['rules'])} rules ({safe} automation-safe) -> {args.output}")


if __name__ == "__main__":
    main()
