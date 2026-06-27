from __future__ import annotations

import pandas as pd

from src.prices.align import PRICE_COLS, PRODUCTS, load_prices


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value * 100):.1f}%"


def price_window_before_news(
    prices: pd.DataFrame,
    news_date: pd.Timestamp,
    window_weeks: int,
) -> str:
    idx = prices.index[prices["date"] < news_date]
    if len(idx) == 0:
        return "Нет истории цен до даты новости."

    window = prices.loc[idx[-window_weeks:] if len(idx) >= window_weeks else idx]
    lines = [
        f"Недельное окно цен ДО новости ({len(window)} нед., акцент на динамику):"
    ]
    for i in range(1, len(window)):
        row = window.iloc[i]
        prev = window.iloc[i - 1]
        days_before = (news_date - pd.Timestamp(row["date"])).days
        parts = []
        for product in PRODUCTS:
            col = PRICE_COLS[product]
            ret = float(row[col]) / float(prev[col]) - 1.0
            parts.append(f"{product} {format_pct(ret)}")
        lines.append(
            f"- {pd.Timestamp(row['date']).strftime('%Y-%m-%d')} "
            f"(за {days_before} дн. до новости): {', '.join(parts)}"
        )
    return "\n".join(lines)


def news_window_before(
    news_df: pd.DataFrame,
    news_date: pd.Timestamp,
    window_days: int,
) -> str:
    start = news_date - pd.Timedelta(days=window_days)
    past = news_df[(news_df["date"] >= start) & (news_df["date"] < news_date)].sort_values("date")
    if past.empty:
        return "Нет новостей в окне до текущей даты."

    lines = [f"Новости за {window_days} дн. ДО текущей (с указанием давности):"]
    for _, row in past.iterrows():
        days_ago = (news_date - pd.Timestamp(row["date"])).days
        lines.append(
            f"- {days_ago} дн. назад | {row['source']} | {row['headline']}"
        )
    return "\n".join(lines)


def build_price_context(prices_path: str, news_date: pd.Timestamp, window_weeks: int) -> str:
    prices = load_prices(prices_path)
    return price_window_before_news(prices, news_date, window_weeks)


def build_news_context(
    news_df: pd.DataFrame,
    news_date: pd.Timestamp,
    window_days: int,
) -> str:
    return news_window_before(news_df, news_date, window_days)
