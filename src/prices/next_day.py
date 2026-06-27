from __future__ import annotations

import pandas as pd

from src.prices.align import PRICE_COLS, PRODUCTS, load_prices


def _direction_from_return(ret: float, flat_threshold: float) -> str:
    if abs(ret) <= flat_threshold:
        return "flat"
    return "up" if ret > 0 else "down"


def compute_next_day_returns(
    news_df: pd.DataFrame,
    prices_path: str,
) -> pd.DataFrame:
    prices = load_prices(prices_path)
    price_cols = [PRICE_COLS[p] for p in PRODUCTS]
    daily = prices.set_index("date")[price_cols].resample("D").ffill()

    rows: list[dict] = []
    for _, news in news_df.iterrows():
        d0 = pd.Timestamp(news["date"]).normalize()
        if d0 not in daily.index:
            continue

        future = prices[prices["date"].dt.normalize() > d0]
        if future.empty:
            continue

        next_quote = future.iloc[0]
        d1 = pd.Timestamp(next_quote["date"]).normalize()

        row = {
            "date": d0,
            "source": news["source"],
            "headline": news["headline"],
            "next_quote_date": d1,
            "days_to_next_quote": int((d1 - d0).days),
        }
        for product in PRODUCTS:
            col = PRICE_COLS[product]
            p0 = float(daily.loc[d0, col])
            p1 = float(next_quote[col])
            if p0 == 0:
                row[f"delta_pct_{product}"] = 0.0
                continue
            row[f"delta_pct_{product}"] = p1 / p0 - 1.0
        rows.append(row)

    return pd.DataFrame(rows)


def directions_from_returns(df: pd.DataFrame, flat_threshold: float) -> pd.DataFrame:
    out = df.copy()
    for product in PRODUCTS:
        out[f"true_dir_{product}"] = out[f"delta_pct_{product}"].apply(
            lambda r: _direction_from_return(float(r), flat_threshold)
        )
    return out


def compute_next_day_ground_truth(
    news_df: pd.DataFrame,
    prices_path: str,
    flat_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Ground truth for short-term move after news.

    Prices are weekly. A literal calendar next-day (D+1) return with forward-fill
    is almost always 0. We therefore compare:
      P0 = quote on news calendar day (daily forward-fill)
      P1 = first weekly quote strictly after the news day
    This is the earliest observable price change after the news (typically 2–7 days).
    """
    returns = compute_next_day_returns(news_df, prices_path)
    return directions_from_returns(returns, flat_threshold)
