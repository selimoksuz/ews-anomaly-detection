"""Run-scoped file logging utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from engine.config_loader import resolve_project_path


def _resolve_level(config: dict) -> int:
    level_name = str(config.get("logging", {}).get("level", "INFO")).strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _build_formatter(config: dict) -> logging.Formatter:
    logging_cfg = config.get("logging", {})
    return logging.Formatter(
        fmt=logging_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
        datefmt=logging_cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )


def get_log_directory(config: dict, category: str) -> Path:
    base_dir = resolve_project_path(
        config.get("logging", {}).get(
            "directory",
            config.get("registry", {}).get("logs_dir", "runtime/logs"),
        )
    )
    path = base_dir / str(category).strip().lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_log_path(config: dict, *, category: str, run_id: str) -> Path:
    return get_log_directory(config, category) / f"{run_id}.log"


@contextmanager
def attach_run_file_logger(config: dict, *, category: str, run_id: str):
    """Attach a run-scoped file handler to the root logger for the context duration."""
    path = get_run_log_path(config, category=category, run_id=run_id)
    level = _resolve_level(config)
    root_logger = logging.getLogger()
    previous_level = root_logger.level

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_build_formatter(config))

    if previous_level == logging.NOTSET or previous_level > level:
        root_logger.setLevel(level)
    root_logger.addHandler(handler)
    try:
        logging.getLogger(__name__).info("Attached %s log handler for run %s at %s", category, run_id, path)
        yield path
    finally:
        logging.getLogger(__name__).info("Closing %s log handler for run %s", category, run_id)
        root_logger.removeHandler(handler)
        handler.close()
        root_logger.setLevel(previous_level)
