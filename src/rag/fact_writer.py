from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "fact_text": {"type": "string"},
        "movement_summary": {
            "type": "object",
            "properties": {
                "urea": {"type": "string"},
                "dap": {"type": "string"},
                "mop": {"type": "string"},
            },
            "required": ["urea", "dap", "mop"],
            "additionalProperties": False,
        },
    },
    "required": ["fact_text", "movement_summary"],
    "additionalProperties": False,
}


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value * 100):.1f}%"


def build_index_text(
    source: str,
    headline: str,
    category: str,
    delta_pct: dict[str, float],
    fact_text: str,
) -> str:
    urea = format_pct(delta_pct["urea"])
    dap = format_pct(delta_pct["dap"])
    mop = format_pct(delta_pct["mop"])
    return (
        f"{source} | {headline}\n"
        f"Категория: {category}\n"
        f"Эффект через 1 нед.: urea {urea}, dap {dap}, mop {mop}\n"
        f"{fact_text}"
    )


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


class FactWriter:
    def __init__(self, client: OpenAI, model: str, config: dict[str, Any]) -> None:
        self.client = client
        self.model = model
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        root = Path(config["project_root"])
        self.system_prompt = (root / "prompts" / "fact_writer.md").read_text(encoding="utf-8")

    def write_fact(
        self,
        source: str,
        headline: str,
        category: str,
        delta_pct: dict[str, float],
        retries: int = 2,
    ) -> dict[str, Any]:
        user_message = (
            f"Источник: {source}\n"
            f"Заголовок: {headline}\n"
            f"Категория: {category}\n"
            f"Изменение цен через 1 неделю:\n"
            f"- urea: {format_pct(delta_pct['urea'])}\n"
            f"- dap: {format_pct(delta_pct['dap'])}\n"
            f"- mop: {format_pct(delta_pct['mop'])}\n"
            "\nСформируй JSON по схеме."
        )
        message = user_message
        last_error: Exception | None = None

        for _ in range(retries + 1):
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
                            "name": "historical_fact",
                            "strict": True,
                            "schema": FACT_SCHEMA,
                        },
                    },
                )
                raw = _parse_json(response.choices[0].message.content or "")
                fact_text = raw["fact_text"]
                return {
                    "fact_text": fact_text,
                    "movement_summary": raw["movement_summary"],
                    "index_text": build_index_text(
                        source, headline, category, delta_pct, fact_text
                    ),
                }
            except Exception as exc:
                last_error = exc
                message = user_message + f"\n\nОшибка ({exc}). Верни только JSON."

        raise RuntimeError(f"FactWriter failed: {last_error}")
