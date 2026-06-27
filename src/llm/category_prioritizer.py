from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from src.llm.llm_meta import usage_from_response

CATEGORIES = [
    "Аммиак",
    "Фосфаты",
    "Калий",
    "Энергоносители",
    "Логистика",
    "Макро/валюта",
    "Спрос/агрорынок",
    "Прочее",
]

OTHER_CATEGORY = "Прочее"

PRIORITIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "reasoning": {"type": "string"},
    },
    "required": ["category", "reasoning"],
    "additionalProperties": False,
}


def load_prioritizer_prompts(config: dict[str, Any]) -> tuple[str, str]:
    root = Path(config["project_root"])
    rel = config.get("prompts", {}).get("streak_predict", "prompts/streak_predict.yaml")
    path = root / rel
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("prioritizer", {})
    system = str(block.get("system", "")).strip()
    user = str(block.get("user", "")).strip()
    if not system or not user:
        raise ValueError(f"Missing prioritizer prompts in {path}")
    return system, user


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _normalize_category(raw: str) -> str:
    value = raw.strip()
    if value in CATEGORIES:
        return value
    lower = value.lower()
    for cat in CATEGORIES:
        if cat.lower() == lower:
            return cat
    return OTHER_CATEGORY


def is_actionable(category: str) -> bool:
    return _normalize_category(category) != OTHER_CATEGORY


class CategoryPrioritizer:
    def __init__(self, client: OpenAI, model: str, config: dict[str, Any]) -> None:
        self.client = client
        self.model = model
        self.config = config
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.system_prompt, self.user_template = load_prioritizer_prompts(config)

    def _build_user_message(self, source: str, headline: str) -> str:
        return self.user_template.format(source=source, headline=headline)

    def _request_llm(
        self,
        user_message: str,
        *,
        use_schema: bool = True,
    ) -> tuple[str, dict[str, int], float]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if use_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "news_category",
                    "strict": True,
                    "schema": PRIORITIZER_SCHEMA,
                },
            }
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError("Empty LLM response")
        return content, usage_from_response(response), elapsed_ms

    def _call_llm_with_meta(
        self,
        source: str,
        headline: str,
        retries: int = 5,
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        user_message = self._build_user_message(source, headline)
        message = user_message
        last_error: Exception | None = None
        attempts = retries + 1
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_ms = 0.0

        for attempt in range(attempts):
            use_schema = attempt < attempts - 2
            try:
                content, usage, elapsed_ms = self._request_llm(message, use_schema=use_schema)
                total_ms += elapsed_ms
                for key in total_usage:
                    total_usage[key] += usage.get(key, 0)
                raw = _parse_json(content)
                category = _normalize_category(str(raw.get("category", OTHER_CATEGORY)))
                reasoning = str(raw.get("reasoning", "")).strip()
                return {"category": category, "reasoning": reasoning}, total_usage, total_ms
            except Exception as exc:
                last_error = exc
                message = user_message + f"\n\nОшибка ({exc}). Верни только JSON."
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Category prioritizer failed: {last_error}")

    def _call_llm(self, source: str, headline: str, retries: int = 5) -> dict[str, Any]:
        result, _, _ = self._call_llm_with_meta(source, headline, retries=retries)
        return result

    def _call_llm_safe(self, source: str, headline: str, retries: int = 7) -> dict[str, Any]:
        try:
            return self._call_llm(source, headline, retries=retries)
        except Exception as exc:
            return {
                "category": OTHER_CATEGORY,
                "reasoning": f"Fallback Прочее: API/parse error ({exc})",
            }

    def predict_with_meta(self, source: str, headline: str) -> tuple[dict[str, Any], dict[str, int], float]:
        try:
            return self._call_llm_with_meta(source, headline)
        except Exception as exc:
            return (
                {
                    "category": OTHER_CATEGORY,
                    "reasoning": f"Fallback Прочее: API/parse error ({exc})",
                },
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                0.0,
            )

    def predict(self, source: str, headline: str) -> dict[str, Any]:
        return self._call_llm_safe(source, headline)

    def predict_batch(
        self,
        items: list[tuple[str, str]],
        max_workers: int = 5,
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        results: list[dict[str, Any] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._call_llm_safe, source, headline, 5): idx
                for idx, (source, headline) in enumerate(items)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pass

        for idx, (source, headline) in enumerate(items):
            if results[idx] is None:
                results[idx] = self._call_llm_safe(source, headline, retries=7)

        return results
