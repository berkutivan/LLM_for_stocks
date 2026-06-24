from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
PIPELINE_DIR = ROOT / "Pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import run_ticker_pipeline
from schemas import AnalysisMode, GraphState, Ticker, Ticker_from_news

PRODUCTS = ("urea", "dap", "mop")
STRATEGY_KEYS = ("buy", "sell", "volatility")
NEUTRAL_PROB = 1 / 3
PRICE_COLUMNS = {
    "urea": "urea_price",
    "dap": "dap_price",
    "mop": "mop_price",
}
DEFAULT_STRATEGY_CSV = ROOT / "right_strategy.csv"
DEFAULT_DYNAMIC_GAIN = 0.5


@dataclass
class SolverOutput:
    news: list[str]
    classes: list[str]
    strategy: dict[str, str]
    latency: float
    token_usage: dict[str, int]


class Solver(ABC):
    """Абстрактный solver: на вход новости за день, на выход — предсказание через get()."""

    @abstractmethod
    def get(self, news_date: str, news: dict[str, str]) -> SolverOutput:
        """Вернуть классы новостей и стратегию на следующий день."""


def _neutral_probs() -> dict[str, float]:
    return {key: NEUTRAL_PROB for key in STRATEGY_KEYS}


def _neutral_ticker() -> Ticker:
    return Ticker(
        Increase_price=NEUTRAL_PROB,
        Decrease_price=NEUTRAL_PROB,
        Volatility_price=NEUTRAL_PROB,
    )


def _default_ticker_from_news() -> Ticker_from_news:
    neutral = _neutral_ticker()
    return Ticker_from_news(
        urea_ticker=neutral,
        dap_ticker=neutral,
        mop_ticker=neutral,
    )


def _normalize_news_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _row_to_ticker_from_news(row: Any) -> Ticker_from_news:
    def make_ticker(prefix: str) -> Ticker:
        return Ticker(
            Increase_price=float(row[f"{prefix}_increase"]),
            Decrease_price=float(row[f"{prefix}_decrease"]),
            Volatility_price=float(row[f"{prefix}_volatility"]),
        )

    return Ticker_from_news(
        urea_ticker=make_ticker("urea"),
        dap_ticker=make_ticker("dap"),
        mop_ticker=make_ticker("mop"),
    )


def _load_predictions_lookup(path: Path) -> dict[tuple[str, str, str], pd.Series]:
    df = pd.read_csv(path)
    lookup: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in df.iterrows():
        key = (
            _normalize_news_date(row["news_date"]),
            str(row["source"]).strip(),
            str(row["headline"]).strip(),
        )
        lookup[key] = row
    return lookup


def _ticker_to_probs(ticker: Ticker) -> dict[str, float]:
    return {
        "buy": ticker.Increase_price,
        "sell": ticker.Decrease_price,
        "volatility": ticker.Volatility_price,
    }


def _strategy_from_probs(
    probs: dict[str, float],
    volatility_weight: float = 1.0,
) -> str:
    scores = {
        "buy": probs["buy"],
        "sell": probs["sell"],
        "volatility": probs["volatility"] * volatility_weight,
    }
    max_val = max(scores.values())
    tied = [key for key in STRATEGY_KEYS if abs(scores[key] - max_val) < 1e-9]
    if len(tied) > 1:
        for preferred in ("buy", "sell", "volatility"):
            if preferred in tied:
                return preferred
    return tied[0]


def _kalman_update(
    prior: dict[str, float],
    measurement: dict[str, float],
    gain: float,
) -> dict[str, float]:
    return {
        key: prior[key] + gain * (measurement[key] - prior[key])
        for key in STRATEGY_KEYS
    }


def dynamic_gain_from_delta(delta: float, alpha: float) -> float:
    """K = 1 / (1 + alpha * delta), delta — доля изменения цены (|ΔP/P|)."""
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return 1.0 / (1.0 + alpha * delta)


def _product_deltas_between_rows(prev: Any, curr: Any) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for product in PRODUCTS:
        price_col = PRICE_COLUMNS[product]
        prev_price = prev[price_col]
        curr_price = curr[price_col]
        if pd.notna(curr_price) and pd.notna(prev_price) and float(prev_price) != 0.0:
            deltas[product] = abs(float(curr_price) / float(prev_price) - 1.0)
    return deltas


def build_daily_deltas(
    strategy_csv: Path | str,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, dict[str, float]]:
    """
    Для каждого календарного дня возвращает |ΔP/P| по продуктам
    между последней доступной ценовой точкой и предыдущей (без lookahead).
    """
    df = pd.read_csv(strategy_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return {}

    delta_at_price_date: dict[date, dict[str, float]] = {}
    for idx in range(1, len(df)):
        curr_day = df.iloc[idx]["date"].date()
        deltas = _product_deltas_between_rows(df.iloc[idx - 1], df.iloc[idx])
        if deltas:
            delta_at_price_date[curr_day] = deltas

    if not delta_at_price_date:
        return {}

    range_start = start or df["date"].min().date()
    range_end = end or df["date"].max().date()
    sorted_price_dates = sorted(delta_at_price_date.keys())

    daily_deltas: dict[str, dict[str, float]] = {}
    pointer = -1
    current_delta: dict[str, float] | None = None
    current = range_start
    while current <= range_end:
        while (
            pointer + 1 < len(sorted_price_dates)
            and sorted_price_dates[pointer + 1] <= current
        ):
            pointer += 1
            current_delta = delta_at_price_date[sorted_price_dates[pointer]]
        if current_delta is not None:
            daily_deltas[current.isoformat()] = dict(current_delta)
        current += timedelta(days=1)

    return daily_deltas


def _aggregate_measurement(state: GraphState) -> dict[str, dict[str, float]]:
    """Сводит тикеры новостей дня в один вектор вероятностей на продукт."""
    if not state.ticker_from_news:
        return {product: _neutral_probs() for product in PRODUCTS}

    measurements: dict[str, dict[str, float]] = {}

    for product in PRODUCTS:
        attr = f"{product}_ticker"
        weighted = {key: 0.0 for key in STRATEGY_KEYS}
        total_weight = 0.0

        for item, priority in zip(state.ticker_from_news, state.prioritased_news):
            ticker = getattr(item, attr)
            weight = max(priority, 1) / 100.0
            probs = _ticker_to_probs(ticker)
            for key in STRATEGY_KEYS:
                weighted[key] += probs[key] * weight
            total_weight += weight

        if total_weight == 0:
            measurements[product] = _neutral_probs()
        else:
            measurements[product] = {
                key: weighted[key] / total_weight for key in STRATEGY_KEYS
            }

    return measurements


class KalmanStrategySolver(Solver):
    """
    Стратегия Калмана: внутренний вектор вероятностей по продуктам
    обновляется при появлении новостей через коэффициент коррекции K:
    x = x + K * (z - x), где z — агрегированный тикер дня.

    Фиксированный K задаётся через gain. Динамический режим: alpha is not None,
    K = 1 / (1 + alpha * delta), delta — |ΔP/P| между i и i-1 для news_date.

    test_mode=True: тикеры и классы читаются из predictions_csv, без LLM.
    """

    def __init__(
        self,
        gain: float = 0.5,
        alpha: float | None = None,
        strategy_csv: Path | str | None = None,
        analysis_mode: AnalysisMode = "standard",
        test_mode: bool = False,
        predictions_csv: Path | str | None = None,
        volatility_weight: float = 1.0,
    ) -> None:
        if alpha is None:
            if not 0.0 <= gain <= 1.0:
                raise ValueError("gain must be between 0 and 1")
            self.gain = gain
            self.alpha = None
            self._daily_deltas: dict[str, dict[str, float]] = {}
        else:
            if not 1.0 <= alpha <= 10.0:
                raise ValueError("alpha must be between 1 and 10")
            self.gain = gain
            self.alpha = alpha
            price_csv = Path(strategy_csv) if strategy_csv is not None else DEFAULT_STRATEGY_CSV
            self._daily_deltas = build_daily_deltas(price_csv)
        if volatility_weight < 0:
            raise ValueError("volatility_weight must be non-negative")
        self.analysis_mode = analysis_mode
        self.test_mode = test_mode
        self.volatility_weight = volatility_weight
        self._belief: dict[str, dict[str, float]] | None = None

        if test_mode:
            if predictions_csv is None:
                raise ValueError("predictions_csv is required when test_mode=True")
            self.predictions_csv = Path(predictions_csv)
            self._predictions_lookup = _load_predictions_lookup(self.predictions_csv)
        else:
            self.predictions_csv = None
            self._predictions_lookup = {}

    def _current_belief(self) -> dict[str, dict[str, float]]:
        if self._belief is None:
            return {product: _neutral_probs() for product in PRODUCTS}
        return self._belief

    def _gain_for_product(self, news_date: str, product: str) -> float:
        if self.alpha is None:
            return self.gain

        day_deltas = self._daily_deltas.get(news_date)
        if day_deltas is None or product not in day_deltas:
            return DEFAULT_DYNAMIC_GAIN

        return dynamic_gain_from_delta(day_deltas[product], self.alpha)

    def _strategies_from_belief(
        self, belief: dict[str, dict[str, float]]
    ) -> dict[str, str]:
        return {
            product: _strategy_from_probs(
                belief[product],
                volatility_weight=self.volatility_weight,
            )
            for product in PRODUCTS
        }

    def _state_from_predictions(
        self, news_date: str, news: dict[str, str]
    ) -> GraphState:
        classified_news: list[str] = []
        prioritased_news: list[int] = []
        ticker_from_news: list[Ticker_from_news] = []
        filtered_news: dict[str, str] = {}
        seen_descriptions: set[str] = set()

        for key, headline in news.items():
            text = headline.strip()
            if not text or text in seen_descriptions:
                continue
            seen_descriptions.add(text)

            source, _, item_headline = key.partition("-")
            lookup_key = (news_date, source.strip(), item_headline.strip())
            row = self._predictions_lookup.get(lookup_key)

            filtered_news[key] = text
            if row is None:
                classified_news.append("Прочее")
                prioritased_news.append(0)
                ticker_from_news.append(_default_ticker_from_news())
                continue

            classified_news.append(str(row["predicted_category"]))
            prioritased_news.append(int(row["priority"]))
            ticker_from_news.append(_row_to_ticker_from_news(row))

        return GraphState.create(
            data=news_date,
            news=filtered_news,
            analysis_mode=self.analysis_mode,
        ).model_copy(
            update={
                "classified_news": classified_news,
                "prioritased_news": prioritased_news,
                "ticker_from_news": ticker_from_news,
            }
        )

    def get(self, news_date: str, news: dict[str, str]) -> SolverOutput:
        if self.test_mode:
            state = self._state_from_predictions(news_date, dict(news))
            latency = 0.0
            token_usage: dict[str, int] = {}
        else:
            state = run_ticker_pipeline(
                data=news_date,
                news=dict(news),
                analysis_mode=self.analysis_mode,
            )
            latency = sum(state.time_use.values())
            token_usage = dict(state.token_use)

        belief = self._current_belief()

        if state.news:
            measurements = _aggregate_measurement(state)
            updated: dict[str, dict[str, float]] = {}
            for product in PRODUCTS:
                updated[product] = _kalman_update(
                    belief[product],
                    measurements[product],
                    self._gain_for_product(news_date, product),
                )
            self._belief = updated
            belief = updated
            strategy = self._strategies_from_belief(belief)
        elif self._belief is None:
            strategy = {product: "volatility" for product in PRODUCTS}
        else:
            strategy = self._strategies_from_belief(belief)

        return SolverOutput(
            news=list(state.news.values()),
            classes=list(state.classified_news),
            strategy=strategy,
            latency=latency,
            token_usage=token_usage,
        )
