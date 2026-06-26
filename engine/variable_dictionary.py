"""Variable dictionary helpers for raw and generated anomaly inputs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from engine.config_loader import PROJECT_ROOT


DEFAULT_DICTIONARY_PATH = PROJECT_ROOT / "config" / "dictionaries.yaml"


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower()


@lru_cache(maxsize=8)
def load_variable_dictionary(path: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path) if path else DEFAULT_DICTIONARY_PATH
    if not resolved.exists():
        return {}
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Variable dictionary must contain a mapping at root: {resolved}")
    return data


def raw_variable_groups(dictionary: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    payload = (dictionary or load_variable_dictionary()).get("raw_variables", {}) or {}
    groups = payload.get("groups", {}) or {}
    return groups if isinstance(groups, dict) else {}


def raw_variable_metadata(dictionary: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group_name, group_payload in raw_variable_groups(dictionary).items():
        if not isinstance(group_payload, dict):
            continue
        variables = group_payload.get("variables", {}) or {}
        source_defaults = group_payload.get("source", {}) or {}
        category = group_payload.get("category", group_name)
        for name, metadata in variables.items():
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {"definition": str(metadata)}
            normalized = _normalize_name(name)
            merged = {
                "group": group_name,
                "category": metadata.get("category", category),
                "source": {**source_defaults, **(metadata.get("source", {}) or {})},
                **metadata,
            }
            merged["name"] = normalized
            result[normalized] = merged
    return result


def generated_variable_metadata(
    dictionary: dict[str, Any] | None = None,
    *,
    enabled_only: bool = False,
) -> dict[str, dict[str, Any]]:
    payload = (dictionary or load_variable_dictionary()).get("generated_variables", {}) or {}
    variables = payload.get("variables", {}) or {}
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(variables, dict):
        return result
    for name, metadata in variables.items():
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"formula": str(metadata)}
        if enabled_only and metadata.get("enabled", True) is False:
            continue
        normalized = _normalize_name(name)
        merged = dict(metadata)
        merged["name"] = normalized
        result[normalized] = merged
    return result


def final_feature_policy(dictionary: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = (dictionary or load_variable_dictionary()).get("final_llm_features", {}) or {}
    return payload if isinstance(payload, dict) else {}


def raw_variable_label_map(dictionary: dict[str, Any] | None = None) -> dict[str, str]:
    result = {}
    for name, metadata in raw_variable_metadata(dictionary).items():
        label = metadata.get("label") or metadata.get("definition")
        if label:
            result[name] = str(label)
    return result


def feature_label_map(dictionary: dict[str, Any] | None = None) -> dict[str, str]:
    result = {}
    for name, metadata in generated_variable_metadata(dictionary).items():
        label = metadata.get("label") or metadata.get("definition")
        if label:
            result[name] = str(label)
    for name, metadata in raw_variable_metadata(dictionary).items():
        label = metadata.get("label") or metadata.get("definition")
        if label:
            result.setdefault(name, str(label))
    return result


def feature_formula_map(dictionary: dict[str, Any] | None = None) -> dict[str, str]:
    result = {}
    for name, metadata in generated_variable_metadata(dictionary).items():
        formula = metadata.get("formula")
        if formula:
            result[name] = str(formula)
    return result


def generated_feature_names(dictionary: dict[str, Any] | None = None, *, enabled_only: bool = True) -> set[str]:
    return set(generated_variable_metadata(dictionary, enabled_only=enabled_only))


def generated_feature_inputs(feature: str, dictionary: dict[str, Any] | None = None) -> list[str]:
    metadata = generated_variable_metadata(dictionary).get(_normalize_name(feature), {})
    inputs = metadata.get("inputs", []) or []
    if isinstance(inputs, str):
        inputs = [inputs]
    return [_normalize_name(item) for item in inputs if str(item).strip()]


def generated_feature_formula(feature: str, dictionary: dict[str, Any] | None = None) -> str | None:
    metadata = generated_variable_metadata(dictionary).get(_normalize_name(feature), {})
    formula = metadata.get("formula")
    return str(formula) if formula else None


def generated_source_columns(dictionary: dict[str, Any] | None = None) -> set[str]:
    result: set[str] = set()
    for name, metadata in generated_variable_metadata(dictionary).items():
        inputs = metadata.get("inputs", []) or []
        if isinstance(inputs, str):
            inputs = [inputs]
        for item in inputs:
            normalized = _normalize_name(item)
            if normalized and normalized != name:
                result.add(normalized)
    return result


def llm_direct_allowed_features(dictionary: dict[str, Any] | None = None) -> set[str]:
    policy = final_feature_policy(dictionary)
    values = policy.get("direct_allowed", []) or []
    return {_normalize_name(item) for item in values if str(item).strip()}


def llm_excluded_feature_names(dictionary: dict[str, Any] | None = None) -> set[str]:
    policy = final_feature_policy(dictionary)
    result = set()
    for key in ("exclude", "forbidden"):
        values = policy.get(key, []) or []
        result.update(_normalize_name(item) for item in values if str(item).strip())
    return result


def final_llm_include_features(dictionary: dict[str, Any] | None = None) -> set[str]:
    policy = final_feature_policy(dictionary)
    values = policy.get("include", []) or []
    return {_normalize_name(item) for item in values if str(item).strip()}


def variable_metadata(name: str, dictionary: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _normalize_name(name)
    generated = generated_variable_metadata(dictionary).get(normalized)
    if generated:
        return generated
    return raw_variable_metadata(dictionary).get(normalized, {})
