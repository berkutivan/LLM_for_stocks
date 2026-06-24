"""Гибрид ticker: перевод полей → FinBERT ×3 (urea/dap/mop) → structured output LLM."""

from __future__ import annotations

import json
import os

from llm.llm_schemas import (
    TickerAgentOutput,
    _record_tokens,
    call_ticker_hybrid_agent,
    get_model_name,
    get_openai_client,
)
from llm.models.connectors import FinBERT, FinBERTSentimentDetailed
from llm.prompt_loader import (
    format_hybrid_log,
    format_prompt,
    format_ticker_finbert_input,
    format_ticker_hybrid_prompts,
    get_ticker_finbert_products,
    get_ticker_translate_fields,
)
from schemas import ModelLog, NewsCategory

DEFAULT_TRANSLATE_MODEL = "gpt-4o-mini"


def get_translate_model_name() -> str:
    return (
        os.environ.get("OPENAI_MODEL_TRANSLATOR", "").strip()
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or DEFAULT_TRANSLATE_MODEL
    )


def _pick_fields(field_names: list[str], values: dict[str, str]) -> dict[str, str]:
    return {name: values[name] for name in field_names}


def _parse_fields_json(raw: str) -> dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("translator: expected JSON object")
    return {str(key): str(value) for key, value in parsed.items()}


def translate_news_fields(
    values: dict[str, str],
    token_use: dict[str, int],
) -> tuple[dict[str, str], list[str]]:
    """Переводит поля ticker (source, news_class, headline) на английский."""
    field_names = get_ticker_translate_fields()
    client = get_openai_client()
    model = get_translate_model_name()
    errors: list[str] = []
    fields = _pick_fields(field_names, values)
    fields_json = json.dumps(fields, ensure_ascii=False, indent=2)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": format_prompt("finbert_translator", "system")},
                {
                    "role": "user",
                    "content": format_prompt(
                        "finbert_translator", "user", fields_json=fields_json
                    ),
                },
            ],
            temperature=0,
        )
    except Exception as exc:
        errors.append(f"translator: {exc}")
        return {}, errors

    usage = completion.usage
    if usage is not None:
        _record_tokens(
            token_use,
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    raw = (completion.choices[0].message.content or "").strip()
    if not raw:
        errors.append("translator: empty translation")
        return {}, errors

    try:
        translated = _parse_fields_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"translator: {exc}")
        return {}, errors

    missing = [
        key for key in field_names if key not in translated or not translated[key].strip()
    ]
    if missing:
        errors.append(f"translator: missing fields: {', '.join(missing)}")
        return {}, errors

    return translated, errors


def _append_log(
    logs: list[ModelLog],
    *,
    node: str,
    model: str,
    input_text: str,
    output_text: str,
) -> None:
    logs.append(
        ModelLog(node=node, model=model, io="input", content=input_text)
    )
    logs.append(
        ModelLog(node=node, model=model, io="output", content=output_text)
    )


def _run_finbert(english: str) -> tuple[FinBERTSentimentDetailed, str, str]:
    finbert = FinBERT()
    response = finbert.responses.parse(
        input=english,
        text_format=FinBERTSentimentDetailed,
    )
    parsed = response.output_parsed
    return parsed, finbert.model_name, parsed.model_dump_json()


def call_ticker_hybrid(
    *,
    source: str,
    news_class: NewsCategory,
    headline: str,
    token_use: dict[str, int],
    node: str = "ticker_from_news_node",
) -> tuple[TickerAgentOutput | None, list[str], list[ModelLog]]:
    """
    Гибрид ticker:
    1) перевести source, news_class, headline;
    2) FinBERT по одному вызову на urea, dap, mop;
    3) structured output LLM с тремя FinBERT JSON.
    """
    logs: list[ModelLog] = []
    errors: list[str] = []

    raw_values = {
        "source": source,
        "news_class": news_class,
        "headline": headline,
    }
    translated, translate_errors = translate_news_fields(raw_values, token_use)
    errors.extend(translate_errors)
    if not translated:
        return None, errors, logs

    finbert_by_product: dict[str, str] = {}
    for product in get_ticker_finbert_products():
        finbert_input = format_ticker_finbert_input(
            source=translated["source"],
            news_class=translated["news_class"],
            headline=translated["headline"],
            product=product,
        )
        try:
            _, finbert_model, finbert_output = _run_finbert(finbert_input)
        except Exception as exc:
            errors.append(f"finbert ({product}): {exc}")
            return None, errors, logs

        finbert_by_product[product] = finbert_output
        _append_log(
            logs,
            node=node,
            model=finbert_model,
            input_text=finbert_input,
            output_text=finbert_output,
        )

    finbert_urea_json = finbert_by_product["urea"]
    finbert_dap_json = finbert_by_product["dap"]
    finbert_mop_json = finbert_by_product["mop"]

    ticker_model = get_model_name("ticker")
    system_prompt, user_prompt = format_ticker_hybrid_prompts(
        source=source,
        news_class=news_class,
        headline=headline,
        finbert_urea_json=finbert_urea_json,
        finbert_dap_json=finbert_dap_json,
        finbert_mop_json=finbert_mop_json,
    )
    agent_input = format_hybrid_log(system_prompt, user_prompt)

    result, agent_errors = call_ticker_hybrid_agent(
        source=source,
        news_class=news_class,
        headline=headline,
        finbert_urea_json=finbert_urea_json,
        finbert_dap_json=finbert_dap_json,
        finbert_mop_json=finbert_mop_json,
        token_use=token_use,
    )
    errors.extend(agent_errors)
    agent_output = (
        json.dumps(result.model_dump(), ensure_ascii=False)
        if result is not None
        else ""
    )
    _append_log(
        logs,
        node=node,
        model=ticker_model,
        input_text=agent_input,
        output_text=agent_output,
    )
    return result, errors, logs
