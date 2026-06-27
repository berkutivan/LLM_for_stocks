from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.llm.llm_meta import merge_usage, usage_from_response
from src.llm.predict_graph import PredictGraphRunner, _route_after_categorize, build_predict_graph
from src.prices.align import PRODUCTS


def _mock_prioritizer(category: str, reasoning: str = "test", ms: float = 120.0):
    prioritizer = MagicMock()
    prioritizer.predict_with_meta.return_value = (
        {"category": category, "reasoning": reasoning},
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        ms,
    )
    return prioritizer


def _mock_predictor(variant: int = 3, ms: float = 450.0):
    predictor = MagicMock()
    prediction = {"reasoning": "pred", "variant": variant}
    for product in PRODUCTS:
        prediction[f"dir_{product}"] = "up"
        prediction[f"days_{product}"] = 0
        prediction[f"weeks_{product}"] = 0
    predictor.predict_with_meta.return_value = (
        prediction,
        {"prompt_tokens": 800, "completion_tokens": 150, "total_tokens": 950},
        ms,
    )
    return predictor


class TestLlmMeta:
    def test_usage_from_response_missing(self):
        assert usage_from_response(object()) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_merge_usage(self):
        merged = merge_usage(
            {
                "a": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "b": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )
        assert merged == {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        }


class TestPredictGraphRouting:
    def test_route_actionable(self):
        assert _route_after_categorize({"actionable": True}) == "predict_next_day"

    def test_route_prochee(self):
        assert _route_after_categorize({"actionable": False}) == "finalize_skip"


class TestPredictGraphSmoke:
    def test_graph_compiles(self):
        graph = build_predict_graph(
            _mock_prioritizer("Аммиак"),
            _mock_predictor(),
            variant=3,
        )
        assert graph is not None

    def test_graph_skips_prochee(self):
        prioritizer = _mock_prioritizer("Прочее")
        predictor = _mock_predictor()
        runner = PredictGraphRunner(
            prioritizer,
            predictor,
            variant=3,
            no_cache=True,
        )
        state = runner.invoke("Reuters", "ESG report", pd.Timestamp("2024-06-01"))

        assert state["category"] == "Прочее"
        assert state["skipped"] is True
        assert state["prediction"] is None
        assert state["step_timings_ms"]["categorize"] == pytest.approx(120.0)
        assert state["token_usage"]["categorize"]["total_tokens"] == 120
        assert "predict_next_day" not in state["step_timings_ms"]
        predictor.predict_with_meta.assert_not_called()

    def test_graph_predicts_for_actionable(self):
        prioritizer = _mock_prioritizer("Аммиак")
        predictor = _mock_predictor()
        runner = PredictGraphRunner(
            prioritizer,
            predictor,
            variant=3,
            no_cache=True,
        )
        state = runner.invoke("Argus", "Gas prices rise", pd.Timestamp("2024-05-01"))

        assert state["actionable"] is True
        assert state["skipped"] is False
        assert state["prediction"] is not None
        assert state["prediction"]["dir_urea"] == "up"
        assert state["step_timings_ms"]["categorize"] == pytest.approx(120.0)
        assert state["step_timings_ms"]["predict_next_day"] == pytest.approx(450.0)
        totals = runner.totals(state)
        assert totals["total_tokens"] == 120 + 950
        summary = runner.format_summary(state)
        assert "tokens=1070" in summary
        assert "categorize=120ms" in summary
        assert "predict_next_day=450ms" in summary

    def test_graph_uses_cache(self):
        cache = {
            "Reuters|||ESG report|||prioritizer": {
                "category": "Прочее",
                "reasoning": "cached",
            }
        }
        prioritizer = _mock_prioritizer("Аммиак")
        runner = PredictGraphRunner(
            prioritizer,
            _mock_predictor(),
            variant=3,
            cache=cache,
            no_cache=False,
            prioritizer_key=lambda s, h: f"{s}|||{h}|||prioritizer",
            prediction_key=lambda s, h, v: f"{s}|||{h}|||v{v}",
        )
        state = runner.invoke("Reuters", "ESG report", pd.Timestamp("2024-06-01"))

        prioritizer.predict_with_meta.assert_not_called()
        assert state["category"] == "Прочее"
        assert state["step_timings_ms"]["categorize"] == 0.0
