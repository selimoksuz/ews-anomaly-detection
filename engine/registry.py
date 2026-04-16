"""File-based run and model registry for champion/challenger workflows."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import time

from engine.config_loader import resolve_project_path


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


class _RegistryLock:
    """Best-effort file lock for registry JSON updates."""

    def __init__(self, lock_path: Path, timeout_seconds: float = 30.0, poll_seconds: float = 0.1):
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.fd: Optional[int] = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for registry lock: {self.lock_path}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


@dataclass
class RunContext:
    run_id: str
    run_type: str
    segment: str
    created_at: str
    run_dir: Path
    artifact_dir: Path
    manifest_path: Path


class RegistryManager:
    """Manage local run registry, model registry, and champion pointers."""

    def __init__(self, config: dict):
        registry_cfg = config.get("registry", {})
        self.registry_dir = resolve_project_path(
            registry_cfg.get("registry_dir", registry_cfg.get("meta_dir", "runtime/registry"))
        )
        self.models_dir = resolve_project_path(
            registry_cfg.get("models_dir", registry_cfg.get("artifacts_dir", "runtime/models"))
        )
        self.run_registry_path = resolve_project_path(
            registry_cfg.get("run_registry_file", self.registry_dir / "run_registry.json")
        )
        self.model_registry_path = Path(
            resolve_project_path(
                registry_cfg.get("model_registry_file", self.registry_dir / "model_registry.json")
            )
        )
        self.champion_registry_path = Path(
            resolve_project_path(
                registry_cfg.get("champion_registry_file", self.registry_dir / "champions.json")
            )
        )
        self.registry_lock_path = resolve_project_path(
            registry_cfg.get("registry_lock_file", self.registry_dir / ".registry.lock")
        )
        self._ensure_layout()

    def start_run(self, run_type: str, segment: str, config: dict, extra: Optional[dict] = None) -> RunContext:
        created_at = _utc_now()
        run_id = f"{run_type}-{segment}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_dir = self.registry_dir / "runs" / run_id
        artifact_dir = self.models_dir / segment / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": run_id,
            "run_type": run_type,
            "segment": segment,
            "created_at": created_at,
            "status": "running",
            "artifact_dir": str(artifact_dir),
            "config_hash": self.compute_config_hash(config),
            "details": extra or {},
        }
        manifest_path = run_dir / "manifest.json"
        with self._registry_lock():
            self._write_json(manifest_path, payload)
            registry = self._read_json(self.run_registry_path, [])
            registry.append(payload)
            self._write_json(self.run_registry_path, registry)

        return RunContext(
            run_id=run_id,
            run_type=run_type,
            segment=segment,
            created_at=created_at,
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
        )

    def finish_run(self, run: RunContext, status: str, summary: Optional[dict] = None):
        summary = summary or {}
        with self._registry_lock():
            manifest = self._read_json(run.manifest_path, {})
            manifest["status"] = status
            manifest["finished_at"] = _utc_now()
            manifest["summary"] = summary
            self._write_json(run.manifest_path, manifest)

            registry = self._read_json(self.run_registry_path, [])
            for item in registry:
                if item["run_id"] == run.run_id:
                    item["status"] = status
                    item["finished_at"] = manifest["finished_at"]
                    item["summary"] = summary
                    break
            self._write_json(self.run_registry_path, registry)

    def register_model(self, record: dict):
        with self._registry_lock():
            registry = self._read_json(self.model_registry_path, [])
            registry.append(record)
            self._write_json(self.model_registry_path, registry)

    def list_models(self, segment: Optional[str] = None) -> list[dict]:
        models = self._read_json(self.model_registry_path, [])
        if segment is None:
            return models
        return [item for item in models if item.get("segment") == segment]

    def get_model(self, model_version: str) -> dict:
        for item in self._read_json(self.model_registry_path, []):
            if item.get("model_version") == model_version:
                return item
        raise KeyError(f"Model version not found: {model_version}")

    def get_latest_candidate(self, segment: str) -> Optional[dict]:
        models = [item for item in self.list_models(segment) if item.get("status") == "candidate"]
        if not models:
            return None
        return sorted(models, key=lambda item: item["created_at"])[-1]

    def get_champion(self, segment: str) -> Optional[dict]:
        champion_map = self._read_json(self.champion_registry_path, {})
        version = champion_map.get(segment)
        if not version:
            return None
        return self.get_model(version)

    def update_model(self, model_version: str, updates: dict) -> dict:
        with self._registry_lock():
            registry = self._read_json(self.model_registry_path, [])
            for item in registry:
                if item.get("model_version") == model_version:
                    item.update(updates)
                    self._write_json(self.model_registry_path, registry)
                    return item
            raise KeyError(f"Model version not found: {model_version}")

    def promote_model(self, segment: str, model_version: str):
        with self._registry_lock():
            champion_map = self._read_json(self.champion_registry_path, {})
            champion_map[segment] = model_version
            self._write_json(self.champion_registry_path, champion_map)

            registry = self._read_json(self.model_registry_path, [])
            for item in registry:
                if item.get("segment") != segment:
                    continue
                item["status"] = "candidate"
                if item.get("model_version") == model_version:
                    item["status"] = "champion"
                    item["promoted_at"] = _utc_now()
            self._write_json(self.model_registry_path, registry)

    def rebuild_run_registry(self) -> list[dict]:
        """Rebuild the run registry from per-run manifest files."""
        manifests = []
        for manifest_path in sorted((self.registry_dir / "runs").glob("*/manifest.json")):
            try:
                manifest = self._read_json(manifest_path, {}, allow_rebuild=False)
            except json.JSONDecodeError:
                continue
            if manifest:
                manifests.append(manifest)
        manifests.sort(key=lambda item: (item.get("created_at", ""), item.get("run_id", "")))
        with self._registry_lock():
            self._write_json(self.run_registry_path, manifests)
        return manifests

    @staticmethod
    def compute_config_hash(config: dict) -> str:
        payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_layout(self):
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.registry_dir / "runs").mkdir(parents=True, exist_ok=True)
        self._ensure_file(self.run_registry_path, [])
        self._ensure_file(self.model_registry_path, [])
        self._ensure_file(self.champion_registry_path, {})

    def _ensure_file(self, path: Path, default):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self._write_json(path, default)

    def _read_json(self, path: Path, default, *, allow_rebuild: bool = True):
        if not path.exists():
            return deepcopy(default)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            if allow_rebuild and path == self.run_registry_path:
                rebuilt = []
                for manifest_path in sorted((self.registry_dir / "runs").glob("*/manifest.json")):
                    try:
                        manifest = self._read_json(manifest_path, {}, allow_rebuild=False)
                    except json.JSONDecodeError:
                        continue
                    if manifest:
                        rebuilt.append(manifest)
                rebuilt.sort(key=lambda item: (item.get("created_at", ""), item.get("run_id", "")))
                self._write_json(self.run_registry_path, rebuilt)
                return rebuilt
            raise

    def _write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _registry_lock(self):
        return _RegistryLock(self.registry_lock_path)
