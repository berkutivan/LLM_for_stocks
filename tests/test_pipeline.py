from __future__ import annotations

import json

import pytest

from conftest import has_llm_credentials
from llm.llm_schemas import get_model_name
from nodes import preprocess_node
from pipeline import run_ticker_pipeline
from schemas import GraphState

requires_llm = pytest.mark.skipif(
    not has_llm_credentials(),
    reason="OPENAI_API_KEY or OPENROUTER_API_KEY is not set",
)


def _print_pipeline_result(state: GraphState) -> None:
    print(f"\nDate: {state.data}")
    print(f"News count: {len(state.news)}")
    print(f"Models: {json.dumps({name: get_model_name(name) for name in ('prioritizer', 'ticker', 'helper')}, ensure_ascii=False)}")

    for index, (key, headline) in enumerate(state.news.items()):
        print(f"\n[{index}] {key}")
        print(f"    headline: {headline}")
        print(f"    class:    {state.classified_news[index]}")
        print(f"    priority: {state.prioritased_news[index]}")
        print(f"    ticker:   {state.ticker_from_news[index].model_dump()}")

    print(f"\ntoken_use: {state.token_use}")
    print(f"time_use:  {state.time_use}")
    if state.error_messages:
        print(f"errors:    {state.error_messages}")


class TestGraphStateFromCsv:
    def test_create_from_market_news(self, single_news_day: tuple[str, dict[str, str]]) -> None:
        day, news = single_news_day
        state = GraphState.create(data=day, news=news)

        assert state.data == day
        assert state.news == news
        assert state.classified_news == []
        assert state.prioritased_news == []
        assert state.ticker_from_news == []


class TestPreprocessFromCsv:
    def test_preprocess_market_news(self, multi_news_day: tuple[str, dict[str, str]]) -> None:
        day, news = multi_news_day
        state = GraphState.create(data=day, news=news)
        result = preprocess_node(state)

        assert len(result.news) <= len(news)
        assert all(description.strip() for description in result.news.values())
        assert "preprocess_node" in result.time_use


class TestPipelineFromCsv:
    @requires_llm
    def test_run_pipeline_single_day(self, single_news_day: tuple[str, dict[str, str]]) -> None:
        day, news = single_news_day
        state = run_ticker_pipeline(data=day, news=news)

        assert len(state.classified_news) == len(state.news)
        assert len(state.prioritased_news) == len(state.news)
        assert len(state.ticker_from_news) == len(state.news)
        assert "pipeline_total" in state.time_use

        _print_pipeline_result(state)

    @requires_llm
    def test_run_pipeline_multi_news_day(
        self, multi_news_day: tuple[str, dict[str, str]]
    ) -> None:
        day, news = multi_news_day
        state = run_ticker_pipeline(data=day, news=news)

        assert len(state.news) == len(news)
        assert len(state.ticker_from_news) == len(news)

        _print_pipeline_result(state)


def _print_model_logs(state: GraphState) -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"\nanalysis_mode: {state.analysis_mode}")
    print(f"model_logs ({len(state.model_logs)}):")
    for index, log in enumerate(state.model_logs, start=1):
        print(f"\n--- log {index}/{len(state.model_logs)} ---")
        print(f"node:  {log.node}")
        print(f"model: {log.model}")
        print(f"{log.io.upper()}:\n{log.content}")


class TestHybridPipeline:
    @requires_llm
    def test_hybrid_single_news_model_logs(
        self, single_news_day: tuple[str, dict[str, str]]
    ) -> None:
        day, all_news = single_news_day
        key, headline = next(iter(all_news.items()))
        news = {key: headline}

        state = run_ticker_pipeline(data=day, news=news, analysis_mode="hybrid")

        assert len(state.news) == 1
        assert len(state.classified_news) == 1
        assert len(state.ticker_from_news) == 1
        assert state.analysis_mode == "hybrid"
        assert len(state.model_logs) == 10

        prioritizer_logs = [
            log for log in state.model_logs if log.node == "prioritizer_node"
        ]
        ticker_logs = [
            log for log in state.model_logs if log.node == "ticker_from_news_node"
        ]
        assert len(prioritizer_logs) == 2
        assert len(ticker_logs) == 8

        for log in state.model_logs:
            assert log.content.strip()

        assert prioritizer_logs[0].io == "input"
        assert prioritizer_logs[1].io == "output"

        finbert_inputs = [
            log.content
            for log in ticker_logs
            if log.io == "input" and "Fertilizer product:" in log.content
        ]
        assert len(finbert_inputs) == 3
        assert "Fertilizer product: urea" in finbert_inputs[0]
        assert "Fertilizer product: dap" in finbert_inputs[1]
        assert "Fertilizer product: mop" in finbert_inputs[2]

        llm_logs = [
            log
            for log in ticker_logs
            if "FinBERT sentiment (urea):" in log.content
            or (log.io == "output" and "urea_ticker" in log.content)
        ]
        assert any(log.io == "input" and "FinBERT sentiment (urea):" in log.content for log in ticker_logs)
        assert any(log.io == "output" and "urea_ticker" in log.content for log in ticker_logs)
        assert len(llm_logs) == 2

        _print_model_logs(state)
    def test_get_model_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_MODEL_PRIORITIZER", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert get_model_name("prioritizer") == "gpt-4o-mini"

        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        assert get_model_name("ticker") == "gpt-4o"

        monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        assert get_model_name("helper") == "deepseek/deepseek-v4-flash"
