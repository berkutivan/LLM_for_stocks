from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI

from src.context.windows import build_news_context, build_price_context
from src.llm.llm_meta import usage_from_response
from src.prices.align import PRODUCTS

STREAK_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "predictions": {
            "type": "object",
            "properties": {
                product: {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["up", "down", "flat"]},
                        "weeks": {"type": "integer", "minimum": 0, "maximum": 7},
                    },
                    "required": ["direction", "weeks"],
                    "additionalProperties": False,
                }
                for product in PRODUCTS
            },
            "required": PRODUCTS,
            "additionalProperties": False,
        },
    },
    "required": ["reasoning", "predictions"],
    "additionalProperties": False,
}

DAYS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "predictions": {
            "type": "object",
            "properties": {
                product: {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["up", "down", "flat"]},
                        "days": {"type": "integer", "minimum": 1, "maximum": 4},
                    },
                    "required": ["direction", "days"],
                    "additionalProperties": False,
                }
                for product in PRODUCTS
            },
            "required": PRODUCTS,
            "additionalProperties": False,
        },
    },
    "required": ["reasoning", "predictions"],
    "additionalProperties": False,
}


def _load_analyst_recommendations(config: dict[str, Any]) -> str:
    root = Path(config["project_root"])
    rel = config.get("prompts", {}).get("analyst", "prompts/system.md")
    path = root / rel
    return path.read_text(encoding="utf-8").strip()


def load_streak_prompts(config: dict[str, Any]) -> dict[int, str]:
    root = Path(config["project_root"])
    rel = config.get("prompts", {}).get("streak_predict", "prompts/streak_predict.yaml")
    path = root / rel
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = data.get("variants", {})
    analyst = _load_analyst_recommendations(config)
    task_prefix = (
        "## Роль и рекомендации аналитика (из system.md)\n\n"
        f"{analyst}\n\n"
        "---\n\n"
        "## Задача предсказания серии\n\n"
    )
    return {
        int(key): task_prefix + str(entry["system"]).strip()
        for key, entry in variants.items()
    }


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


def _normalize_streak_prediction(raw: dict[str, Any], variant: int = 1) -> dict[str, Any]:
    preds = raw.get("predictions")
    if not isinstance(preds, dict):
        raise ValueError("Missing predictions in LLM response")
    reasoning = raw.get("reasoning", "")
    result: dict[str, Any] = {"reasoning": reasoning}
    for product in PRODUCTS:
        p = preds.get(product, {})
        direction = p.get("direction", "flat")
        if direction not in ("up", "down", "flat"):
            direction = "flat"
        if variant >= 3:
            days = int(p.get("days", 1) or 1)
            days = max(1, min(4, days))
            result[f"dir_{product}"] = direction
            result[f"days_{product}"] = days
            result[f"weeks_{product}"] = 0
        else:
            weeks = int(p.get("weeks", 0) or 0)
            if direction == "flat":
                weeks = 0
            result[f"dir_{product}"] = direction
            result[f"weeks_{product}"] = max(0, min(7, weeks))
            result[f"days_{product}"] = result[f"weeks_{product}"] * 7
    return result


def _flat_fallback(reason: str, variant: int) -> dict[str, Any]:
    result: dict[str, Any] = {"reasoning": reason, "variant": variant}
    for product in PRODUCTS:
        result[f"dir_{product}"] = "flat"
        if variant >= 3:
            result[f"weeks_{product}"] = 0
            result[f"days_{product}"] = 1
        else:
            result[f"weeks_{product}"] = 0
            result[f"days_{product}"] = 0
    return result


class StreakPredictor:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        config: dict[str, Any],
        retriever: Any,
        news_df: pd.DataFrame,
        prices_path: str,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config
        self.retriever = retriever
        self.news_df = news_df.sort_values("date").reset_index(drop=True)
        self.prices_path = prices_path
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.prompts = load_streak_prompts(config)
        ctx = config.get("context", {})
        self.window_weeks = ctx.get("window_weeks", 4)
        self.window_days = ctx.get("window_days", 28)

    def _build_user_message(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
    ) -> str:
        precedents = self.retriever.retrieve(source, headline)
        parts = [
            f"Дата новости: {news_date.strftime('%Y-%m-%d')}",
            f"Источник: {source}",
            f"Заголовок: {headline}",
        ]
        if category:
            parts.extend(["", f"## Категория новости\n{category}"])
        parts.extend([
            "",
            "## RAG-прецеденты (source + headline → серии после новости)",
            "\n\n".join(precedents) if precedents else "Нет прецедентов.",
        ])

        if variant >= 2:
            price_ctx = build_price_context(self.prices_path, news_date, self.window_weeks)
            parts.extend(["", "## Окно цен до новости", price_ctx])

        if variant >= 3:
            news_ctx = build_news_context(self.news_df, news_date, self.window_days)
            parts.extend(["", "## Окно прошлых новостей", news_ctx])

        parts.append("\nВерни JSON по схеме.")
        return "\n".join(parts)

    def _request_llm(
        self,
        system_prompt: str,
        user_message: str,
        *,
        use_schema: bool = True,
        variant: int = 1,
    ) -> tuple[str, dict[str, int], float]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if use_schema:
            schema = DAYS_SCHEMA if variant >= 3 else STREAK_SCHEMA
            schema_name = "days_prediction" if variant >= 3 else "streak_prediction"
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
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
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
        retries: int = 5,
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        system_prompt = self.prompts[variant]
        user_message = self._build_user_message(
            source, headline, news_date, variant, category=category
        )
        message = user_message
        last_error: Exception | None = None
        attempts = retries + 1
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_ms = 0.0

        for attempt in range(attempts):
            use_schema = attempt < attempts - 2
            try:
                content, usage, elapsed_ms = self._request_llm(
                    system_prompt, message, use_schema=use_schema, variant=variant
                )
                total_ms += elapsed_ms
                for key in total_usage:
                    total_usage[key] += usage.get(key, 0)
                raw = _parse_json(content)
                result = _normalize_streak_prediction(raw, variant=variant)
                result["variant"] = variant
                return result, total_usage, total_ms
            except Exception as exc:
                last_error = exc
                message = user_message + f"\n\nОшибка ({exc}). Верни только JSON."
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Variant {variant} failed: {last_error}")

    def _call_llm(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
        retries: int = 5,
    ) -> dict[str, Any]:
        result, _, _ = self._call_llm_with_meta(
            source, headline, news_date, variant, category=category, retries=retries
        )
        return result

    def _call_llm_safe(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
        retries: int = 7,
    ) -> dict[str, Any]:
        try:
            return self._call_llm(
                source, headline, news_date, variant, category=category, retries=retries
            )
        except Exception as exc:
            return _flat_fallback(
                f"Fallback flat: API/parse error ({exc})",
                variant,
            )

    def predict_with_meta(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        try:
            return self._call_llm_with_meta(
                source, headline, news_date, variant, category=category
            )
        except Exception as exc:
            return (
                _flat_fallback(f"Fallback flat: API/parse error ({exc})", variant),
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                0.0,
            )

    def predict(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int,
        category: str | None = None,
    ) -> dict[str, Any]:
        return self._call_llm_safe(source, headline, news_date, variant, category=category)

    def predict_variants(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variants: list[int],
        category: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        return {
            v: self.predict(source, headline, news_date, v, category=category) for v in variants
        }

    def predict_batch(
        self,
        items: list[tuple[str, str, pd.Timestamp, str | None]],
        variant: int,
        max_workers: int = 5,
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        results: list[dict[str, Any] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._call_llm_safe, s, h, d, variant, c, 5): i
                for i, (s, h, d, c) in enumerate(items)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pass

        for i, (s, h, d, c) in enumerate(items):
            if results[i] is None:
                results[i] = self._call_llm_safe(s, h, d, variant, category=c, retries=7)

        return results
