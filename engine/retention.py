"""Retention utilities for logs, run metadata, and artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
