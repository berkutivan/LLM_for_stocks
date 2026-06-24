from __future__ import annotations

from time import perf_counter
from typing import Callable

from nodes import preprocess_node, prioritizer_node, ticker_from_news_node
from schemas import AnalysisMode, GraphState

NodeFn = Callable[[GraphState], GraphState]

DEFAULT_NODES: tuple[NodeFn, ...] = (
    preprocess_node,
    prioritizer_node,
    ticker_from_news_node,
)


class TickerPipeline:
    """
    Пайплайн генерации тикеров по новостям за день.

    preprocess → prioritizer → ticker_from_news
    """

    def __init__(
        self,
        nodes: tuple[NodeFn, ...] | None = None,
        analysis_mode: AnalysisMode = "standard",
    ) -> None:
        self._nodes = nodes or DEFAULT_NODES
        self._analysis_mode = analysis_mode

    def run(self, data: str, news: dict[str, str]) -> GraphState:
        """
        Запуск пайплайна.

        Args:
            data: Дата новостей (YYYY-MM-DD).
            news: Словарь source-headline -> описание/заголовок.

        Returns:
            GraphState с classified_news, prioritased_news и ticker_from_news.
        """
        state = GraphState.create(
            data=data,
            news=news,
            analysis_mode=self._analysis_mode,
        )
        started = perf_counter()

        for node in self._nodes:
            state = node(state)

        time_use = dict(state.time_use)
        time_use["pipeline_total"] = perf_counter() - started
        return state.model_copy(update={"time_use": time_use})


def run_ticker_pipeline(
    data: str,
    news: dict[str, str],
    analysis_mode: AnalysisMode = "standard",
) -> GraphState:
    """Сокращённый вызов TickerPipeline().run(...)."""
    return TickerPipeline(analysis_mode=analysis_mode).run(data=data, news=news)
