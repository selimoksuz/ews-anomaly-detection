"""Retention utilities for logs, run metadata, and artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


class RetentionManager:
    """Apply filesystem retention policies configured for the project."""

    def __init__(self, config: dict):
        self.config = config
        self.registry_cfg = config.get("registry", {})
        self.retention_cfg = config.get("retention", {})
        self.logs_dir = Path(self.registry_cfg.get("logs_dir", "logs"))
        self.meta_dir = Path(self.registry_cfg.get("meta_dir", "meta"))
        self.artifacts_dir = Path(self.registry_cfg.get("artifacts_dir", "artifacts"))
        self.run_registry_path = Path(self.registry_cfg.get("run_registry_file", self.meta_dir / "run_registry.json"))
        self.model_registry_path = Path(self.registry_cfg.get("model_registry_file", self.meta_dir / "model_registry.json"))
        self.champion_registry_path = Path(self.registry_cfg.get("champion_registry_file", self.meta_dir / "champions.json"))
        self.registry_lock_path = Path(self.registry_cfg.get("registry_lock_file", self.meta_dir / ".registry.lock"))
        self.monitoring_dir = Path(self.config.get("monitoring", {}).get("directory", self.meta_dir / "monitoring"))

    def cleanup(self) -> dict:
        deleted = {
            "logs": self._cleanup_files(self.logs_dir, int(self.retention_cfg.get("logs_days", 14))),
            "run_manifests": self._cleanup_directories(
                self.meta_dir / "runs",
                int(self.retention_cfg.get("run_manifests_days", 30)),
            ),
            "artifacts": self._cleanup_directories(
                self.artifacts_dir,
                int(self.retention_cfg.get("artifacts_days", 60)),
            ),
        }
        return deleted

    def reset_runtime_state(self) -> dict:
        """Remove all local runtime outputs and recreate empty registry files."""
        deleted = {
            "logs": self._clear_directory(self.logs_dir),
            "artifacts": self._clear_directory(self.artifacts_dir),
            "runs": self._clear_directory(self.meta_dir / "runs"),
            "monitoring": self._clear_directory(self.monitoring_dir),
            "registry_files": 0,
        }

        for path, payload in (
            (self.run_registry_path, []),
            (self.model_registry_path, []),
            (self.champion_registry_path, {}),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            deleted["registry_files"] += 1

        if self.registry_lock_path.exists():
            self.registry_lock_path.unlink(missing_ok=True)

        (self.meta_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        return deleted

    def _cleanup_files(self, directory: Path, max_age_days: int) -> int:
        if not directory.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        deleted = 0
        for item in directory.iterdir():
            if not item.is_file():
                continue
            modified = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                item.unlink(missing_ok=True)
                deleted += 1
        return deleted

    def _cleanup_directories(self, directory: Path, max_age_days: int) -> int:
        if not directory.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        deleted = 0
        for item in directory.iterdir():
            if not item.is_dir():
                continue
            modified = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
            if modified >= cutoff:
                continue
            self._delete_tree(item)
            deleted += 1
        return deleted

    def _delete_tree(self, path: Path):
        for item in path.iterdir():
            if item.is_dir():
                self._delete_tree(item)
            else:
                item.unlink(missing_ok=True)
        path.rmdir()

    def _clear_directory(self, directory: Path) -> int:
        if not directory.exists():
            return 0
        deleted = 0
        for item in list(directory.iterdir()):
            if item.is_dir():
                self._delete_tree(item)
            else:
                item.unlink(missing_ok=True)
            deleted += 1
        return deleted
