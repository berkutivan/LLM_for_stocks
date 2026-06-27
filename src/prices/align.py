from __future__ import annotations

import pandas as pd

PRODUCTS = ["urea", "dap", "mop"]
PRICE_COLS = {"urea": "urea_price", "dap": "dap_price", "mop": "mop_price"}


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def align_news_to_prices(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    lag_weeks: int = 1,
) -> pd.DataFrame:
    prices = prices_df.copy()
    price_dates = prices["date"].tolist()
    rows: list[dict] = []

    for _, news in news_df.iterrows():
        news_date = pd.Timestamp(news["date"])
        p0_idx = prices.index[prices["date"] >= news_date]
        if len(p0_idx) == 0:
            continue
        i0 = int(p0_idx[0])
        i1 = i0 + lag_weeks
        if i1 >= len(prices):
            continue

        row = {
            "date": news_date,
            "source": news["source"],
            "headline": news["headline"],
            "category": news["category"],
            "p0_date": prices.loc[i0, "date"],
            "p1_date": prices.loc[i1, "date"],
        }
        for product in PRODUCTS:
            col = PRICE_COLS[product]
            p0 = float(prices.loc[i0, col])
            p1 = float(prices.loc[i1, col])
            row[f"delta_pct_{product}"] = p1 / p0 - 1.0
            row[f"abs_delta_{product}"] = abs(p1 / p0 - 1.0)
        rows.append(row)

    return pd.DataFrame(rows)
