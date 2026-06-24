from __future__ import annotations

from time import perf_counter

from llm.finbert_hybrid import call_ticker_hybrid
from llm.llm_schemas import call_prioritizer_agent, call_prioritizer_agent_with_logs, call_ticker_agent
from schemas import GraphState, NewsCategory, Ticker, Ticker_from_news


def _parse_news_key(key: str) -> tuple[str, str]:
    source, _, headline = key.partition("-")
    return source, headline


def _neutral_ticker() -> Ticker:
    return Ticker(
        Increase_price=1 / 3,
        Decrease_price=1 / 3,
        Volatility_price=1 / 3,
    )


def _default_ticker_from_news() -> Ticker_from_news:
    neutral = _neutral_ticker()
    return Ticker_from_news(
        urea_ticker=neutral,
        dap_ticker=neutral,
        mop_ticker=neutral,
    )


def preprocess_node(state: GraphState) -> GraphState:
    """
    Оставляет в news только новости с непустым описанием (заголовком).
    Дубликаты по тексту описания удаляются.
    """
    started = perf_counter()
    filtered: dict[str, str] = {}
    seen_descriptions: set[str] = set()

    for key, description in state.news.items():
        text = description.strip()
        if not text:
            continue
        if text in seen_descriptions:
            continue
        seen_descriptions.add(text)
        filtered[key] = text

    time_use = dict(state.time_use)
    time_use["preprocess_node"] = perf_counter() - started
    return state.model_copy(update={"news": filtered, "time_use": time_use})


def prioritizer_node(state: GraphState) -> GraphState:
    """
    Классифицирует и приоритизирует каждую новость через LLM.
    Классы: Аммиак, Фосфаты, Калий, Энергоносители, Логистика,
    Макро/валюта, Спрос/агрорынок, Прочее.
    """
    started = perf_counter()
    classified_news: list[NewsCategory] = []
    prioritased_news: list[int] = []
    token_use = dict(state.token_use)
    error_messages = list(state.error_messages)
    model_logs = list(state.model_logs)

    for key, headline in state.news.items():
        source, _ = _parse_news_key(key)
        if state.analysis_mode == "hybrid":
            result, errors, logs = call_prioritizer_agent_with_logs(
                source=source,
                headline=headline,
                token_use=token_use,
            )
            model_logs.extend(logs)
        else:
            result, errors = call_prioritizer_agent(
                source=source,
                headline=headline,
                token_use=token_use,
            )
        error_messages.extend(errors)

        if result is None:
            classified_news.append("Прочее")
            prioritased_news.append(0)
            error_messages.append(
                f"prioritizer_node: fallback defaults for news '{key}'"
            )
            continue

        classified_news.append(result.category)
        prioritased_news.append(result.priority)

    time_use = dict(state.time_use)
    time_use["prioritizer_node"] = perf_counter() - started
    return state.model_copy(
        update={
            "classified_news": classified_news,
            "prioritased_news": prioritased_news,
            "token_use": token_use,
            "time_use": time_use,
            "error_messages": error_messages,
            "model_logs": model_logs,
        }
    )


def ticker_from_news_node(state: GraphState) -> GraphState:
    """Переводит каждую новость в ticker_from_news через LLM (класс + источник + заголовок)."""
    started = perf_counter()
    ticker_from_news: list[Ticker_from_news] = []
    token_use = dict(state.token_use)
    error_messages = list(state.error_messages)
    model_logs = list(state.model_logs)

    news_items = list(state.news.items())
    for index, (key, headline) in enumerate(news_items):
        source, _ = _parse_news_key(key)
        news_class = (
            state.classified_news[index]
            if index < len(state.classified_news)
            else "Прочее"
        )

        if state.analysis_mode == "hybrid":
            result, errors, logs = call_ticker_hybrid(
                source=source,
                news_class=news_class,
                headline=headline,
                token_use=token_use,
            )
            model_logs.extend(logs)
        else:
            result, errors = call_ticker_agent(
                source=source,
                news_class=news_class,
                headline=headline,
                token_use=token_use,
            )
        error_messages.extend(errors)

        if result is None:
            ticker_from_news.append(_default_ticker_from_news())
            error_messages.append(
                f"ticker_from_news_node: fallback neutral ticker for news '{key}'"
            )
            continue

        ticker_from_news.append(result.ticker)

    time_use = dict(state.time_use)
    time_use["ticker_from_news_node"] = perf_counter() - started
    return state.model_copy(
        update={
            "ticker_from_news": ticker_from_news,
            "token_use": token_use,
            "time_use": time_use,
            "error_messages": error_messages,
            "model_logs": model_logs,
        }
    )
