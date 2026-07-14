"""模型产物发现与元数据管理：仅通过清单文件定位活跃模型。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ModelRegistry:
    """管理心脏与卒中两个独立模型的活跃清单。"""

    def __init__(self, model_dir: str, manifest_path: str, feature_columns=None):
        self.model_dir = Path(model_dir)
        self.manifest_path = Path(manifest_path)
        self.feature_columns = list(feature_columns or [])

    def load_active_models(self) -> dict:
        """只读加载已登记模型，清单不存在时明确返回未就绪。"""
        if not self.manifest_path.exists():
            raise FileNotFoundError("未找到模型清单，请先完成训练。")

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError("模型清单无法读取，请重新训练。") from exc
        if manifest.get("version") != 2:
            raise FileNotFoundError("模型清单版本过旧，请重新训练。")
        if self.feature_columns and manifest.get("feature_columns") != self.feature_columns:
            raise FileNotFoundError("模型清单与当前特征契约不一致，请重新训练。")

        models = manifest.get("models", {})
        if not isinstance(models, dict):
            raise FileNotFoundError("模型清单格式错误，请重新训练。")
        if not all(isinstance(models.get(name), str) and models.get(name) for name in ("heart", "stroke")):
            raise FileNotFoundError("模型清单缺少心脏事件或卒中模型，请重新训练。")
        resolved = {
            name: str(self._resolve_artifact_path(path))
            for name, path in models.items()
        }
        if all(Path(resolved.get(n, "")).exists() for n in ("heart", "stroke")):
            return resolved

        raise FileNotFoundError("活跃模型文件缺失，请重新训练。")

    def register(self, models: dict, metrics: dict) -> dict:
        """写入模型清单，记录路径、指标和特征契约。"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": self.feature_columns,
            "strategy": "two_xgboost_models_with_isotonic_calibration",
            "models": {
                name: self._store_artifact_path(path) for name, path in models.items()
            },
            "metrics": metrics,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _store_artifact_path(self, path: str) -> str:
        artifact = Path(path).resolve()
        try:
            return artifact.relative_to(self.manifest_path.parent.resolve()).as_posix()
        except ValueError:
            raise ValueError("模型文件必须位于模型注册表目录内。") from None

    def _resolve_artifact_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise FileNotFoundError("模型文件路径为空，请重新训练。")
        artifact = Path(path)
        if not artifact.is_absolute():
            artifact = self.manifest_path.parent / artifact
        return artifact.resolve()
