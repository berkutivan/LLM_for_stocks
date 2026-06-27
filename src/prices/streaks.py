from __future__ import annotations

from typing import Any

import pandas as pd

from src.prices.align import PRICE_COLS, PRODUCTS, load_prices

Direction = str  # "up" | "down" | "flat"


def _classify_weekly_return(ret: float, flat_threshold: float) -> Direction:
    if abs(ret) <= flat_threshold:
        return "flat"
    return "up" if ret > 0 else "down"


def _price_index_on_or_after(prices: pd.DataFrame, news_date: pd.Timestamp) -> int | None:
    idx = prices.index[prices["date"] >= news_date]
    if len(idx) == 0:
        return None
    return int(idx[0])


def compute_day_streak_for_product(
    prices: pd.DataFrame,
    news_date: pd.Timestamp,
    product: str,
    max_days: int = 4,
    flat_threshold: float = 0.01,
) -> dict[str, Any]:
    """Consecutive weekly regimes (up/down/flat) after news, counted as days 1–4."""
    i0 = _price_index_on_or_after(prices, news_date)
    if i0 is None or i0 + 1 >= len(prices):
        return {"direction": "flat", "days": 0, "weekly_returns": []}

    col = PRICE_COLS[product]
    weekly_returns: list[float] = []
    for k in range(1, max_days + 1):
        if i0 + k >= len(prices):
            break
        p_prev = float(prices.loc[i0 + k - 1, col])
        p_curr = float(prices.loc[i0 + k, col])
        weekly_returns.append(p_curr / p_prev - 1.0)

    if not weekly_returns:
        return {"direction": "flat", "days": 0, "weekly_returns": []}

    direction = _classify_weekly_return(weekly_returns[0], flat_threshold)
    streak = 1
    for r in weekly_returns[1:]:
        if _classify_weekly_return(r, flat_threshold) == direction:
            streak += 1
        else:
            break

    return {
        "direction": direction,
        "days": min(streak, max_days),
        "weekly_returns": weekly_returns,
    }


def compute_all_day_streaks(
    prices: pd.DataFrame,
    news_date: pd.Timestamp,
    max_days: int = 4,
    flat_threshold: float = 0.01,
) -> dict[str, dict[str, Any]]:
    return {
        product: compute_day_streak_for_product(
            prices, news_date, product, max_days, flat_threshold
        )
        for product in PRODUCTS
    }


def day_streaks_to_fact_text(streaks: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    labels = {"urea": "urea", "dap": "dap", "mop": "mop"}
    dir_ru = {"up": "рост", "down": "падение", "flat": "боковик (flat)"}
    for product in PRODUCTS:
        s = streaks[product]
        if s["days"] == 0:
            parts.append(f"{labels[product]}: нет серии (0 дн.)")
        else:
            parts.append(
                f"{labels[product]}: {s['days']} дн. подряд {dir_ru[s['direction']]}"
            )
    return "; ".join(parts)


def compute_streak_for_product(
    prices: pd.DataFrame,
    news_date: pd.Timestamp,
    product: str,
    max_weeks: int = 7,
    flat_threshold: float = 0.01,
) -> dict[str, Any]:
    i0 = _price_index_on_or_after(prices, news_date)
    if i0 is None or i0 + 1 >= len(prices):
        return {"direction": "flat", "weeks": 0, "days": 0, "weekly_returns": []}

    col = PRICE_COLS[product]
    weekly_returns: list[float] = []
    for k in range(1, max_weeks + 1):
        if i0 + k >= len(prices):
            break
        p_prev = float(prices.loc[i0 + k - 1, col])
        p_curr = float(prices.loc[i0 + k, col])
        weekly_returns.append(p_curr / p_prev - 1.0)

    if not weekly_returns:
        return {"direction": "flat", "weeks": 0, "days": 0, "weekly_returns": []}

    first = weekly_returns[0]
    if abs(first) <= flat_threshold:
        return {"direction": "flat", "weeks": 0, "days": 0, "weekly_returns": weekly_returns}

    direction: Direction = "up" if first > 0 else "down"
    streak = 1
    for r in weekly_returns[1:]:
        if direction == "up" and r > flat_threshold:
            streak += 1
        elif direction == "down" and r < -flat_threshold:
            streak += 1
        else:
            break

    return {
        "direction": direction,
        "weeks": streak,
        "days": streak * 7,
        "weekly_returns": weekly_returns,
    }


def compute_all_streaks(
    prices: pd.DataFrame,
    news_date: pd.Timestamp,
    max_weeks: int = 7,
    flat_threshold: float = 0.01,
) -> dict[str, dict[str, Any]]:
    return {
        product: compute_streak_for_product(
            prices, news_date, product, max_weeks, flat_threshold
        )
        for product in PRODUCTS
    }


def streaks_to_fact_text(streaks: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    labels = {"urea": "urea", "dap": "dap", "mop": "mop"}
    dir_ru = {"up": "рост", "down": "падение", "flat": "без серии"}
    for product in PRODUCTS:
        s = streaks[product]
        if s["direction"] == "flat" or s["weeks"] == 0:
            parts.append(f"{labels[product]}: без серии направленного движения")
        else:
            parts.append(
                f"{labels[product]}: {s['weeks']} нед. подряд {dir_ru[s['direction']]} "
                f"(~{s['days']} дн.)"
            )
    return "; ".join(parts)


def build_index_text(source: str, headline: str, fact_text: str) -> str:
    return f"{source} | {headline}\n{fact_text}"


def compute_streaks_for_news_df(
    news_df: pd.DataFrame,
    prices_path: str,
    max_weeks: int = 7,
    flat_threshold: float = 0.01,
    *,
    mode: str = "weeks",
    max_days: int = 4,
) -> pd.DataFrame:
    prices = load_prices(prices_path)
    rows: list[dict] = []
    for _, news in news_df.iterrows():
        news_date = pd.Timestamp(news["date"])
        if mode == "days":
            streaks = compute_all_day_streaks(prices, news_date, max_days, flat_threshold)
            fact_text = day_streaks_to_fact_text(streaks)
        else:
            streaks = compute_all_streaks(prices, news_date, max_weeks, flat_threshold)
            fact_text = streaks_to_fact_text(streaks)
        rows.append(
            {
                "date": news_date,
                "source": news["source"],
                "headline": news["headline"],
                "streaks": streaks,
                "fact_text": fact_text,
                "index_text": build_index_text(news["source"], news["headline"], fact_text),
            }
        )
    return pd.DataFrame(rows)


def compute_day_streaks_for_news_df(
    news_df: pd.DataFrame,
    prices_path: str,
    max_days: int = 4,
    flat_threshold: float = 0.01,
) -> pd.DataFrame:
    return compute_streaks_for_news_df(
        news_df,
        prices_path,
        flat_threshold=flat_threshold,
        mode="days",
        max_days=max_days,
    )
