"""敏感字段脱敏工具，供预览和随访接口复用。"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


_SENSITIVE_EXACT = {
    "resident_id",
    "person_id",
    "id_card",
    "identity_card",
    "phone",
    "mobile",
    "telephone",
    "name",
    "real_name",
    "address",
    "home_address",
    "email",
    "contact",
}


def is_sensitive_column(column: str) -> bool:
    normalized = str(column).strip().lower()
    if normalized in _SENSITIVE_EXACT:
        return True
    return any(token in normalized for token in ("id_card", "identity", "phone", "mobile"))


def mask_sensitive_value(column: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    normalized = str(column).strip().lower()
    if "phone" in normalized or "mobile" in normalized or "telephone" in normalized:
        return text[:3] + "****" + text[-4:] if len(text) >= 7 else "已脱敏"
    if normalized in {"name", "real_name"}:
        return (text[:1] + "**") if text else "已脱敏"
    if "address" in normalized or normalized in {"contact", "email"}:
        return "已脱敏"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"R-{digest}"


def mask_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    masked = dataframe.copy()
    for column in masked.columns:
        if is_sensitive_column(column):
            masked[column] = masked[column].map(lambda value: mask_sensitive_value(column, value))
    return masked


def mask_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: mask_sensitive_value(key, value) if is_sensitive_column(key) else value
            for key, value in record.items()
        }
        for record in records
    ]
