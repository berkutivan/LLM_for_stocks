from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Callable, Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.llm.category_prioritizer import CategoryPrioritizer, is_actionable
from src.llm.llm_meta import merge_usage


class PredictGraphState(TypedDict, total=False):
    source: str
    headline: str
    news_date: str
    variant: int

    category: str | None
    category_reasoning: str | None
    actionable: bool

    prediction: dict[str, Any] | None
    skipped: bool
    skip_reason: str | None

    step_timings_ms: Annotated[dict[str, float], operator.or_]
    token_usage: Annotated[dict[str, dict[str, int]], operator.or_]
    total_elapsed_ms: float

    errors: Annotated[list[str], operator.add]


def _route_after_categorize(
    state: PredictGraphState,
) -> Literal["predict_next_day", "finalize_skip"]:
    if state.get("actionable"):
        return "predict_next_day"
    return "finalize_skip"


def build_predict_graph(
    prioritizer: CategoryPrioritizer,
    predictor: Any,
    variant: int,
    *,
    cache: dict[str, dict] | None = None,
    no_cache: bool = False,
    prioritizer_key: Callable[[str, str], str] | None = None,
    prediction_key: Callable[[str, str, int], str] | None = None,
):
    cache_store = cache if cache is not None else {}

    def categorize_node(state: PredictGraphState) -> dict[str, Any]:
        source = state["source"]
        headline = state["headline"]
        pkey = prioritizer_key(source, headline) if prioritizer_key else f"{source}|||{headline}|||prioritizer"

        if not no_cache and pkey in cache_store:
            cached = cache_store[pkey]
            category = cached["category"]
            return {
                "category": category,
                "category_reasoning": cached.get("reasoning", ""),
                "actionable": is_actionable(category),
                "step_timings_ms": {"categorize": 0.0},
                "token_usage": {
                    "categorize": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                },
            }

        result, usage, elapsed_ms = prioritizer.predict_with_meta(source, headline)
        cache_store[pkey] = result
        category = result["category"]
        return {
            "category": category,
            "category_reasoning": result.get("reasoning", ""),
            "actionable": is_actionable(category),
            "step_timings_ms": {"categorize": elapsed_ms},
            "token_usage": {"categorize": usage},
        }

    def predict_next_day_node(state: PredictGraphState) -> dict[str, Any]:
        source = state["source"]
        headline = state["headline"]
        news_date = pd.Timestamp(state["news_date"])
        vkey = (
            prediction_key(source, headline, variant)
            if prediction_key
            else f"{source}|||{headline}|||v{variant}"
        )

        if not no_cache and vkey in cache_store:
            cached_pred = cache_store[vkey]
            return {
                "prediction": cached_pred,
                "skipped": False,
                "step_timings_ms": {"predict_next_day": 0.0},
                "token_usage": {
                    "predict_next_day": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                },
            }

        result, usage, elapsed_ms = predictor.predict_with_meta(
            source,
            headline,
            news_date,
            variant,
            category=state.get("category"),
        )
        cache_store[vkey] = result
        return {
            "prediction": result,
            "skipped": False,
            "step_timings_ms": {"predict_next_day": elapsed_ms},
            "token_usage": {"predict_next_day": usage},
        }

    def finalize_skip_node(state: PredictGraphState) -> dict[str, Any]:
        return {
            "prediction": None,
            "skipped": True,
            "skip_reason": "category=Прочее",
        }

    def finalize_node(state: PredictGraphState) -> dict[str, Any]:
        timings = state.get("step_timings_ms", {})
        return {"total_elapsed_ms": sum(timings.values())}

    graph = StateGraph(PredictGraphState)
    graph.add_node("categorize", categorize_node)
    graph.add_node("predict_next_day", predict_next_day_node)
    graph.add_node("finalize_skip", finalize_skip_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "categorize")
    graph.add_conditional_edges(
        "categorize",
        _route_after_categorize,
        {
            "predict_next_day": "predict_next_day",
            "finalize_skip": "finalize_skip",
        },
    )
    graph.add_edge("predict_next_day", "finalize")
    graph.add_edge("finalize_skip", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


class PredictGraphRunner:
    def __init__(
        self,
        prioritizer: CategoryPrioritizer,
        predictor: Any,
        variant: int,
        *,
        cache: dict[str, dict] | None = None,
        no_cache: bool = False,
        prioritizer_key: Callable[[str, str], str] | None = None,
        prediction_key: Callable[[str, str, int], str] | None = None,
    ) -> None:
        self.variant = variant
        self.graph = build_predict_graph(
            prioritizer,
            predictor,
            variant,
            cache=cache,
            no_cache=no_cache,
            prioritizer_key=prioritizer_key,
            prediction_key=prediction_key,
        )

    def invoke(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
    ) -> PredictGraphState:
        t0 = time.perf_counter()
        state: PredictGraphState = self.graph.invoke(
            {
                "source": source,
                "headline": headline,
                "news_date": news_date.strftime("%Y-%m-%d"),
                "variant": self.variant,
                "step_timings_ms": {},
                "token_usage": {},
                "errors": [],
            }
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        state["total_elapsed_ms"] = max(state.get("total_elapsed_ms", 0.0), wall_ms)
        return state

    @staticmethod
    def totals(state: PredictGraphState) -> dict[str, int]:
        return merge_usage(state.get("token_usage", {}))

    @staticmethod
    def format_summary(state: PredictGraphState) -> str:
        totals = PredictGraphRunner.totals(state)
        timings = state.get("step_timings_ms", {})
        timing_str = ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
        return (
            f"elapsed={state.get('total_elapsed_ms', 0.0):.0f}ms "
            f"[{timing_str}] "
            f"tokens={totals['total_tokens']} "
            f"(prompt={totals['prompt_tokens']}, completion={totals['completion_tokens']})"
        )
