from __future__ import annotations

import json
import os
from typing import Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from llm.prioritizer_rag import format_prioritizer_examples, retrieve_prioritizer_examples
from llm.prompt_loader import format_hybrid_log, format_prompt, format_ticker_hybrid_prompts
from schemas import ModelLog, NewsCategory, Ticker_from_news

DEFAULT_MODEL = "gpt-4o-mini"
AgentName = Literal["prioritizer", "ticker", "helper"]

T = TypeVar("T", bound=BaseModel)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")


_load_env()


def get_model_name(agent: AgentName) -> str:
    env_by_agent = {
        "prioritizer": "OPENAI_MODEL_PRIORITIZER",
        "ticker": "OPENAI_MODEL_TICKER",
        "helper": "OPENAI_MODEL_HELPER",
    }
    specific = os.environ.get(env_by_agent[agent], "").strip()
    if specific:
        return specific
    return (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def get_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )


def get_base_url() -> str | None:
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("OPENROUTER_BASE_URL", "").strip()
    )
    if base_url:
        return base_url
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "https://openrouter.ai/api/v1"
    return None


class PrioritizerOutput(BaseModel):
    category: NewsCategory = Field(description="News category")
    priority: int = Field(ge=0, le=100, description="Market impact priority 0-100")


class TickerAgentOutput(BaseModel):
    ticker: Ticker_from_news = Field(description="Probabilistic tickers for urea, dap, mop")


def get_openai_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "API key is not set. Add OPENAI_API_KEY or OPENROUTER_API_KEY to .env "
            "(see env.example)."
        )
    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _record_tokens(
    token_use: dict[str, int],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    token_use[model] = token_use.get(model, 0) + prompt_tokens + completion_tokens


def _parse_structured_response(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: type[T],
    token_use: dict[str, int],
) -> tuple[T | None, str, list[str]]:
    errors: list[str] = []
    raw_response = ""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format,
    )
    message = completion.choices[0].message
    usage = completion.usage
    if usage is not None:
        _record_tokens(
            token_use,
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    if message.parsed is not None:
        return message.parsed, message.content or "", errors

    raw_response = message.content or ""
    if message.refusal:
        errors.append(message.refusal)
    else:
        errors.append("Structured output parsing returned empty result")
    return None, raw_response, errors


def call_structured_agent(
    *,
    agent_name: str,
    response_format: type[T],
    system_prompt: str,
    user_prompt: str,
    model: str,
    token_use: dict[str, int],
) -> tuple[T | None, list[str]]:
    """Call LLM with structured output; on parse failure invoke helper model."""
    client = get_openai_client()
    errors: list[str] = []

    try:
        parsed, raw_response, primary_errors = _parse_structured_response(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            token_use=token_use,
        )
        if parsed is not None:
            return parsed, errors

        errors.extend(primary_errors)
    except (ValidationError, ValueError) as exc:
        raw_response = str(exc)
        errors.append(f"{agent_name}: {exc}")
    except Exception as exc:
        raw_response = str(exc)
        errors.append(f"{agent_name}: {exc}")

    helper_model = get_model_name("helper")
    helper_system = format_prompt("parse_helper", "system")
    helper_user = format_prompt(
        "parse_helper",
        "user",
        agent_name=agent_name,
        error=errors[-1] if errors else "unknown error",
        raw_response=raw_response,
        schema_description=json.dumps(
            response_format.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        ),
    )

    try:
        parsed, _, helper_errors = _parse_structured_response(
            client=client,
            model=helper_model,
            system_prompt=helper_system,
            user_prompt=helper_user,
            response_format=response_format,
            token_use=token_use,
        )
        if parsed is not None:
            return parsed, errors

        errors.extend(helper_errors)
        errors.append(f"{agent_name}: helper model failed to repair structured output")
    except Exception as exc:
        errors.append(f"{agent_name}: helper model error: {exc}")

    return None, errors


def call_prioritizer_agent(
    *,
    source: str,
    headline: str,
    token_use: dict[str, int],
    rag_examples: str | None = None,
) -> tuple[PrioritizerOutput | None, list[str]]:
    model = get_model_name("prioritizer")
    system_prompt = format_prompt("prioritizer", "system")
    if rag_examples is None:
        retrieved = retrieve_prioritizer_examples(source=source, headline=headline)
        rag_examples = format_prioritizer_examples(retrieved)
    user_prompt = format_prompt(
        "prioritizer",
        "user",
        source=source,
        headline=headline,
        rag_examples=rag_examples,
    )
    return call_structured_agent(
        agent_name="prioritizer",
        response_format=PrioritizerOutput,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        token_use=token_use,
    )


def call_prioritizer_agent_with_logs(
    *,
    source: str,
    headline: str,
    token_use: dict[str, int],
    node: str = "prioritizer_node",
    rag_examples: str | None = None,
) -> tuple[PrioritizerOutput | None, list[str], list[ModelLog]]:
    """Стандартный prioritizer со логированием input/output."""
    model = get_model_name("prioritizer")
    if rag_examples is None:
        retrieved = retrieve_prioritizer_examples(source=source, headline=headline)
        rag_examples = format_prioritizer_examples(retrieved)
    system_prompt = format_prompt("prioritizer", "system")
    user_prompt = format_prompt(
        "prioritizer",
        "user",
        source=source,
        headline=headline,
        rag_examples=rag_examples,
    )
    agent_input = format_hybrid_log(system_prompt, user_prompt)

    result, errors = call_prioritizer_agent(
        source=source,
        headline=headline,
        token_use=token_use,
        rag_examples=rag_examples,
    )
    agent_output = (
        json.dumps(result.model_dump(), ensure_ascii=False)
        if result is not None
        else ""
    )
    logs = [
        ModelLog(node=node, model=model, io="input", content=agent_input),
        ModelLog(node=node, model=model, io="output", content=agent_output),
    ]
    return result, errors, logs


def call_ticker_agent(
    *,
    source: str,
    news_class: NewsCategory,
    headline: str,
    token_use: dict[str, int],
) -> tuple[TickerAgentOutput | None, list[str]]:
    model = get_model_name("ticker")
    system_prompt = format_prompt("ticker_from_news", "system")
    user_prompt = format_prompt(
        "ticker_from_news",
        "user",
        source=source,
        news_class=news_class,
        headline=headline,
    )
    return call_structured_agent(
        agent_name="ticker_from_news",
        response_format=TickerAgentOutput,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        token_use=token_use,
    )


def call_ticker_hybrid_agent(
    *,
    source: str,
    news_class: NewsCategory,
    headline: str,
    finbert_urea_json: str,
    finbert_dap_json: str,
    finbert_mop_json: str,
    token_use: dict[str, int],
) -> tuple[TickerAgentOutput | None, list[str]]:
    model = get_model_name("ticker")
    system_prompt, user_prompt = format_ticker_hybrid_prompts(
        source=source,
        news_class=news_class,
        headline=headline,
        finbert_urea_json=finbert_urea_json,
        finbert_dap_json=finbert_dap_json,
        finbert_mop_json=finbert_mop_json,
    )
    return call_structured_agent(
        agent_name="ticker_from_news",
        response_format=TickerAgentOutput,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        token_use=token_use,
    )
