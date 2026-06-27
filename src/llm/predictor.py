from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from src.data.split import temporal_split
from src.rules.hints import CATEGORIES, analyze_headline, format_hints

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "reasoning": {"type": "string"},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "scores": {
            "type": "object",
            "properties": {
                "urea": {"type": "number", "minimum": -1, "maximum": 1},
                "dap": {"type": "number", "minimum": -1, "maximum": 1},
                "mop": {"type": "number", "minimum": -1, "maximum": 1},
            },
            "required": ["urea", "dap", "mop"],
            "additionalProperties": False,
        },
    },
    "required": ["category", "reasoning", "importance", "scores"],
    "additionalProperties": False,
}


def _load_system_prompt(project_root: Path) -> str:
    return (project_root / "prompts" / "system.md").read_text(encoding="utf-8")


def select_few_shot(news_df: pd.DataFrame, train_ratio: float, n: int = 3) -> list[dict[str, str]]:
    train, _ = temporal_split(news_df, train_ratio)

    examples: list[dict[str, str]] = []
    prochee = train[train["category"] == "Прочее"].head(1)
    thematic = train[train["category"] != "Прочее"].head(n - len(prochee))
    for part in [thematic, prochee]:
        for _, row in part.iterrows():
            examples.append(
                {
                    "source": row["source"],
                    "headline": row["headline"],
                    "category": row["category"],
                }
            )
    return examples[:n]


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


def _clamp_score(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _normalize_prediction(raw: dict[str, Any]) -> dict[str, Any]:
    scores = raw["scores"]
    return {
        "category": raw["category"],
        "reasoning": raw["reasoning"],
        "importance": max(0.0, min(1.0, float(raw["importance"]))),
        "score_urea": _clamp_score(scores["urea"]),
        "score_dap": _clamp_score(scores["dap"]),
        "score_mop": _clamp_score(scores["mop"]),
    }


class NewsPredictor:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        config: dict[str, Any],
        retriever: Any,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config
        self.retriever = retriever
        self.few_shot_examples = few_shot_examples or []
        self.system_prompt = _load_system_prompt(Path(config["project_root"]))
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]

    def _build_user_message(self, source: str, headline: str) -> str:
        hints = analyze_headline(source, headline)
        precedents = self.retriever.retrieve(
            source,
            headline,
            suggested_category=hints.suggested_category,
        )
        parts = [
            f"Источник: {source}",
            f"Заголовок: {headline}",
            "",
            "## Pre-analysis hints",
            format_hints(hints),
            "",
            "## Исторические прецеденты (похожие новости из train)",
            "\n\n".join(precedents) if precedents else "Нет близких прецедентов.",
        ]
        if self.few_shot_examples:
            parts.append("\n## Примеры разметки (few-shot, без scores)")
            for ex in self.few_shot_examples:
                parts.append(f"- [{ex['source']}] {ex['headline']} → {ex['category']}")
        parts.append("\nВерни JSON по схеме.")
        return "\n".join(parts)

    def _apply_post_filter(self, source: str, headline: str, result: dict[str, Any]) -> dict[str, Any]:
        hints = analyze_headline(source, headline)
        if hints.likely_prochee and result["category"] != "Прочее" and result["importance"] < 0.3:
            result = dict(result)
            result["category"] = "Прочее"
            result["score_urea"] = 0.0
            result["score_dap"] = 0.0
            result["score_mop"] = 0.0
            result["importance"] = min(result["importance"], 0.1)
            result["reasoning"] = result["reasoning"] + " [post-filter: likely_Прочее]"
        return result

    def _call_llm(self, source: str, headline: str, user_message: str, retries: int = 2) -> dict[str, Any]:
        last_error: Exception | None = None
        message = user_message

        for _attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": message},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "news_prediction",
                            "strict": True,
                            "schema": RESPONSE_SCHEMA,
                        },
                    },
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("Empty LLM response")
                raw = _parse_json(content)
                result = _normalize_prediction(raw)
                return self._apply_post_filter(source, headline, result)
            except Exception as exc:
                last_error = exc
                message = user_message + f"\n\nПредыдущий ответ невалиден ({exc}). Верни только JSON."

        raise RuntimeError(f"Failed after {retries + 1} attempts: {last_error}")

    def predict(self, source: str, headline: str, retries: int = 2) -> dict[str, Any]:
        user_message = self._build_user_message(source, headline)
        return self._call_llm(source, headline, user_message, retries=retries)

    def predict_batch(
        self,
        items: list[tuple[str, str]],
        max_workers: int = 10,
        retries: int = 2,
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        messages = [(source, headline, self._build_user_message(source, headline)) for source, headline in items]
        results: list[dict[str, Any] | None] = [None] * len(messages)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._call_llm, source, headline, user_message, retries): idx
                for idx, (source, headline, user_message) in enumerate(messages)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                source, headline, user_message = messages[idx]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = self._call_llm(
                        source, headline, user_message, retries=retries + 2
                    )

        return [r for r in results if r is not None]
