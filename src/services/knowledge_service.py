"""本地、可审核的健康建议规则知识库。"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class KnowledgeService:
    """按危险因素和风险等级检索固定建议，不调用外部模型。"""

    _cache = {}
    _lock = Lock()
    max_automatic_suggestions = 6

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
            if not active_factors.intersection(factors):
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
                }
            )

        suggestions = sorted(
            candidates, key=lambda item: item.get("priority", 0), reverse=True
        )[: self.max_automatic_suggestions]

        if not suggestions:
            suggestions.append(
                {
                    "rule_id": "general-follow-up",
                    "text": "建议保持规律作息，定期监测关键健康指标，并在需要时咨询专业人员。",
                    "source": "项目通用健康管理模板",
                    "version": content.get("version", "fallback"),
                }
            )
        return {
            "risk_level": risk_level,
            "knowledge_version": content.get("version", "fallback"),
            "suggestions": suggestions,
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
            factors.add("exercise")
        return factors
