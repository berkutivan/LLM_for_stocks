"""Загрузка и форматирование промптов из prompts.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


@lru_cache
def load_prompts() -> dict:
    with PROMPTS_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_prompt_node(*parts: str) -> str:
    node: object = load_prompts()
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Prompt not found: {'.'.join(parts)}")
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(f"Prompt not found: {'.'.join(parts)}")
    return node


def format_prompt(*parts: str, **kwargs: str) -> str:
    template = _resolve_prompt_node(*parts)
    return template.format(**kwargs) if kwargs else template


def format_hybrid_log(system: str, user: str) -> str:
    return format_prompt("hybrid", "log_format", system=system, user=user)


def get_ticker_hybrid_config() -> dict:
    return load_prompts()["hybrid"]["ticker"]


def get_ticker_translate_fields() -> list[str]:
    return list(get_ticker_hybrid_config()["translate_fields"])


def get_ticker_finbert_products() -> list[str]:
    return list(get_ticker_hybrid_config()["finbert_products"])


def format_ticker_finbert_input(**fields: str) -> str:
    finbert_key = get_ticker_hybrid_config()["finbert_prompt"]
    return format_prompt(finbert_key, "user", **fields).strip()


def format_ticker_hybrid_prompts(**kwargs: str) -> tuple[str, str]:
    agent_cfg = get_ticker_hybrid_config()["agent"]
    system = format_prompt(agent_cfg["system"], "system")
    user = format_prompt(agent_cfg["user"], "user", **kwargs)
    return system, user
