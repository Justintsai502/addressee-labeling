"""Tiny config loader (YAML with env-var expansion for api keys)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml  # PyYAML; only needed when you actually run against real models

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_env(cfg)


def _resolve_env(obj: Any) -> Any:
    """Replace any string value of the form ${ENV_VAR} with os.environ[ENV_VAR]."""
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    return obj
