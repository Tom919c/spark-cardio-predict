"""Model artifact discovery and metadata management for the dual-risk platform."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ModelRegistry:
    """Keeps active heart and stroke artifacts independent of browser input."""

    def __init__(self, model_dir: str, manifest_path: str, feature_columns=None):
        self.model_dir = Path(model_dir)
        self.manifest_path = Path(manifest_path)
        self.feature_columns = list(feature_columns or [])

    def load_active_models(self) -> dict:
        if not self.manifest_path.exists():
            raise FileNotFoundError("No active model manifest was found. Train models first.")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 2:
            raise FileNotFoundError("Model manifest version is outdated. Retrain the models.")
        if self.feature_columns and manifest.get("feature_columns") != self.feature_columns:
            raise FileNotFoundError("Model manifest does not match current feature columns.")

        models = manifest.get("models", {})
        resolved_models = {
            name: str(self._resolve_artifact_path(path))
            for name, path in models.items()
        }
        if all(Path(resolved_models.get(name, "")).exists() for name in ("heart", "stroke")):
            return resolved_models

        raise FileNotFoundError("Active heart and stroke model files are missing.")

    def register(self, models: dict, metrics: dict) -> dict:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": self.feature_columns,
            "strategy": "two_random_forests_with_isotonic_calibration",
            "models": {
                name: self._store_artifact_path(path) for name, path in models.items()
            },
            "metrics": metrics,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _store_artifact_path(self, path: str) -> str:
        artifact = Path(path).resolve()
        try:
            return artifact.relative_to(self.manifest_path.parent.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Model artifacts must live under the managed model directory.") from exc

    def _resolve_artifact_path(self, path: str) -> Path:
        artifact = Path(path)
        if not artifact.is_absolute():
            artifact = self.manifest_path.parent / artifact
        return artifact.resolve()
