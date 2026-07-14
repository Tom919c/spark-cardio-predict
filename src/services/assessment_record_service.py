"""个人风险评估记录服务：负责匿名归属、持久化与安全摘要。"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from uuid import uuid4

from src.dao.database_factory import create_database_dao


class AssessmentRecordService:
    """在不保存客户端原始标识的前提下管理个人评估历史。"""

    def __init__(self, config, database_dao=None):
        self.config = config
        self.database_dao = database_dao or create_database_dao(config)
        key = config.get("ASSESSMENT_HASH_KEY") or config.get("SECRET_KEY")
        if not key:
            raise RuntimeError("缺少 ASSESSMENT_HASH_KEY 或 SECRET_KEY，无法安全保存评估记录。")
        self._hash_key = str(key).encode("utf-8")

    def create(self, client_id, sample, result):
        """保存一次评估；未提供客户端标识时生成一个可由调用方保存的标识。"""
        normalized_client_id, generated = self.resolve_client_id(client_id)
        risk_level = result.get("risk_level") or {}
        record = self.database_dao.create_assessment(
            {
                "assessment_id": f"asm_{uuid4().hex}",
                "client_hash": self.hash_client_id(normalized_client_id),
                "model_version": self._model_version(),
                "knowledge_version": result.get("intervention_knowledge_version"),
                "heart_probability": result.get("heart_predicted_probability", 0),
                "stroke_probability": result.get("stroke_predicted_probability", 0),
                "risk_level_code": risk_level.get("code", 1),
                "risk_level_name": risk_level.get("name", "I级：健康"),
                "final_category": result.get("final_category", 0),
                "input_json": self._json_dump(sample),
                "result_json": self._json_dump(result),
            }
        )
        response = self.to_summary(record)
        if generated:
            response["client_id"] = normalized_client_id
        return response

    def list_for_client(self, client_id, limit=10):
        client_hash = self.hash_client_id(self.require_client_id(client_id))
        safe_limit = max(1, min(int(limit), 50))
        return [
            self.to_summary(record)
            for record in self.database_dao.list_assessments(client_hash, safe_limit)
        ]

    def get_for_client(self, assessment_id, client_id):
        client_hash = self.hash_client_id(self.require_client_id(client_id))
        record = self.database_dao.get_assessment(str(assessment_id).strip())
        # 对“不存在”和“不属于当前客户端”返回相同结果，避免泄露记录是否存在。
        if not record or not hmac.compare_digest(record["client_hash"], client_hash):
            raise FileNotFoundError("未找到该评估记录。")
        return self.to_summary(record)

    def hash_client_id(self, client_id):
        return hmac.new(
            self._hash_key,
            self.require_client_id(client_id).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def resolve_client_id(client_id):
        if client_id is None or not str(client_id).strip():
            return secrets.token_urlsafe(24), True
        value = str(client_id).strip()
        AssessmentRecordService._validate_client_id(value)
        return value, False

    @staticmethod
    def require_client_id(client_id):
        if client_id is None or not str(client_id).strip():
            raise ValueError("缺少 X-Client-ID 请求头。")
        value = str(client_id).strip()
        AssessmentRecordService._validate_client_id(value)
        return value

    @staticmethod
    def _validate_client_id(value):
        if len(value) < 8 or len(value) > 256:
            raise ValueError("X-Client-ID 长度必须在 8 到 256 个字符之间。")

    @staticmethod
    def _json_dump(value):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def to_summary(record):
        """仅返回列表和详情页需要的字段，不暴露哈希、输入或完整模型解释。"""
        return {
            "assessment_id": record["assessment_id"],
            "model_version": record.get("model_version"),
            "knowledge_version": record.get("knowledge_version"),
            "heart_probability": float(record.get("heart_probability", 0)),
            "stroke_probability": float(record.get("stroke_probability", 0)),
            "risk_level": {
                "code": int(record.get("risk_level_code", 1)),
                "name": record.get("risk_level_name", "I级：健康"),
            },
            "final_category": int(record.get("final_category", 0)),
            "created_at": record.get("created_at"),
        }

    def _model_version(self):
        manifest_path = self.config.get("MODEL_MANIFEST_PATH")
        if not manifest_path:
            return "unknown"
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return "unknown"
        return str(
            manifest.get("model_version")
            or manifest.get("updated_at")
            or manifest.get("version")
            or "unknown"
        )
