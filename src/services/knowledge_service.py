"""本地、可审核的健康建议规则知识库。"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class KnowledgeService:
    """按危险因素和风险等级检索固定建议，不调用外部模型。"""

    _cache = {}
    _lock = Lock()
    max_automatic_suggestions = 3

    _factor_priority = {
        "hypertension": 70,
        "smoker": 60,
        "diabetes": 50,
        "cholesterol": 40,
        "bmi": 30,
        "exercise": 20,
        "alcohol": 10,
    }
    _friendly_actions = {
        "hypertension": "从今天开始记录早晚血压；如已在用药，请按医嘱坚持服用，不要自行停药。",
        "smoker": "先定一个明确的戒烟日期，并告诉家人；需要时可到戒烟门诊寻求帮助。",
        "diabetes": "规律记录血糖，少喝含糖饮料，并按医嘱复查血糖和糖化血红蛋白。",
        "cholesterol": "下一餐先少一点油炸和肥肉，多选蔬菜、全谷物、鱼类或豆制品。",
        "bmi": "先从每天少一份高热量零食、饭后步行 20 分钟开始，逐步管理体重。",
        "exercise": "从每天步行 20 分钟开始，循序渐进；运动时如有胸闷、胸痛或明显不适请立即停止。",
        "alcohol": "本周先安排无酒日，逐步减少饮酒次数和饮酒量，不用饮酒来预防心血管病。",
    }
    _follow_up_guidance = {
        1: "保持现有健康习惯，建议每 12 个月复查一次血压、血脂和血糖。",
        2: "建议在 6 个月内复查血压、血脂和血糖，并根据结果调整生活方式。",
        3: "建议建立健康记录，并在 3 个月内完成一次随访复查。",
        4: "建议尽快预约医疗机构进行专业评估，并在 1 个月内复查。",
        5: "建议在 2 周内前往医疗机构完成专业评估；这不是急诊判断。",
    }
    emergency_warning = (
        "如出现持续胸痛、呼吸困难、单侧肢体无力、口角歪斜或言语不清，"
        "请立即拨打 120。"
    )

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return {"version": "fallback", "rules": []}
        stat = self.path.stat()
        cache_key = str(self.path.resolve())
        version = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] == version:
                return cached[1]
            try:
                content = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("规则知识库无法读取，请检查 JSON 格式。") from exc
            if not isinstance(content, dict) or not isinstance(content.get("rules"), list):
                raise ValueError("规则知识库格式错误，缺少 rules 数组。")
            if not all(isinstance(rule, dict) for rule in content["rules"]):
                raise ValueError("规则知识库格式错误，rules 必须由对象组成。")
            self._cache[cache_key] = (version, content)
            return content

    def build_plan(self, sample, risk_level):
        level_code = risk_level.get("code", 1) if isinstance(risk_level, dict) else int(risk_level)
        active_factors = self._active_factors(sample)
        if level_code >= 4:
            active_factors.add("risk_level")
        content = self.load()
        candidates = []
        used = set()
        for rule in content.get("rules", []):
            if rule.get("enabled", True) is False:
                continue
            # Only explicitly safe rules can become automatic advice. Missing
            # metadata remains compatible with the original small rule format.
            if rule.get("automation_safe", True) is not True:
                continue
            factors = rule.get("auto_factors") or [rule.get("factor")]
            matched_factors = active_factors.intersection(factors)
            if not matched_factors:
                continue
            if level_code not in rule.get("risk_levels", []):
                continue
            rule_id = rule.get("rule_id")
            if not rule_id or rule_id in used:
                continue
            used.add(rule_id)
            candidates.append(
                {
                    "rule_id": rule_id,
                    "text": rule.get("advice", ""),
                    "source": rule.get("source", "本地规则知识库"),
                    "version": rule.get("version", content.get("version", "unknown")),
                    "domain": rule.get("domain"),
                    "rule_class": rule.get("rule_class"),
                    "automation_safe": True,
                    "review_mode": rule.get("review_mode", "auto"),
                    "target_population": rule.get("target_population"),
                    "boundary_notes": rule.get(
                        "boundary_notes", rule.get("contraindication", "")
                    ),
                    "source_doc": rule.get("source_doc"),
                    "source_section": rule.get("source_section"),
                    "evidence_level": rule.get("evidence_level"),
                    "priority": rule.get("priority", 0),
                    "matched_factors": sorted(matched_factors),
                }
            )

        follow_up_candidates = [
            item for item in candidates if item.get("rule_class") == "follow_up"
        ]
        action_candidates = [
            item for item in candidates if item.get("rule_class") != "follow_up"
        ]

        actions = []
        for factor in sorted(
            active_factors - {"risk_level"},
            key=lambda name: self._factor_priority.get(name, 0),
            reverse=True,
        ):
            matches = [
                item
                for item in action_candidates
                if factor in (item.get("matched_factors") or [])
            ]
            source_item = max(matches, key=lambda item: item.get("priority", 0)) if matches else None
            if source_item is None and factor != "bmi":
                continue
            action = dict(source_item or {})
            action.update(
                {
                    "rule_id": action.get("rule_id", "bmi-weight-management"),
                    "text": self._friendly_actions[factor],
                    "domain": factor,
                    "source": action.get("source", "项目审核版基础健康管理规则"),
                    "version": action.get("version", content.get("version", "fallback")),
                    "automation_safe": True,
                }
            )
            actions.append(action)
            if len(actions) >= self.max_automatic_suggestions:
                break

        follow_up = self._follow_up_guidance.get(level_code, self._follow_up_guidance[1])
        follow_up_item = None
        if follow_up_candidates:
            follow_up_item = max(
                follow_up_candidates, key=lambda item: item.get("priority", 0)
            )
            follow_up_item = {**follow_up_item, "text": follow_up}

        if not actions:
            actions.append(
                {
                    "rule_id": "general-health-maintenance",
                    "text": "继续保持规律作息和适量活动，并定期记录血压、血脂和血糖。",
                    "domain": "general",
                    "source": "项目通用健康管理模板",
                    "version": content.get("version", "fallback"),
                    "automation_safe": True,
                }
            )

        suggestions = ([follow_up_item] if follow_up_item else []) + actions
        factor_names = {
            "hypertension": "血压",
            "smoker": "吸烟",
            "diabetes": "血糖",
            "cholesterol": "血脂",
            "bmi": "体重",
            "exercise": "活动量",
            "alcohol": "饮酒",
        }
        focus = [factor_names[item["domain"]] for item in actions if item["domain"] in factor_names]
        focus_summary = (
            "您现在最需要关注的是" + "、".join(focus) + "。"
            if focus
            else "目前建议以保持健康习惯和定期复查为主。"
        )
        return {
            "risk_level": risk_level,
            "knowledge_version": content.get("version", "fallback"),
            "suggestions": suggestions,
            "focus_summary": focus_summary,
            "actions": actions,
            "follow_up": follow_up,
            "emergency_warning": self.emergency_warning,
        }

    @staticmethod
    def _active_factors(sample):
        factors = set()
        if int(sample.get("hypertension", 0)) == 1:
            factors.add("hypertension")
        if int(sample.get("cholesterol", 1)) >= 2:
            factors.add("cholesterol")
        if int(sample.get("diabetes", 0)) == 1:
            factors.add("diabetes")
        if int(sample.get("smoker", 0)) >= 1:
            factors.add("smoker")
        if int(sample.get("alcohol", 0)) == 1:
            factors.add("alcohol")
        if int(sample.get("exercise", 1)) == 0:
            factors.add("exercise")
        if float(sample.get("bmi", 0)) >= 24:
            factors.add("bmi")
        return factors
