from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")

    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is not set in .env")

    config["llm"]["api_key"] = api_key
    config["llm"]["model"] = model
    config["project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], relative: str) -> Path:
    return PROJECT_ROOT / relative
