"""Корреляция важности новостей (LLM priority) с абсолютными изменениями цен."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEWS_PATH = ROOT / "Results" / "market_to_ticker.csv"
PRICES_PATH = ROOT / "new_prices.csv"

PRODUCTS = ("urea", "dap", "mop")
PRICE_COLS = {"urea": "urea_price", "dap": "dap_price", "mop": "mop_price"}

LAG1_METRICS = [
    ("abs_ret_1_urea", "urea"),
    ("abs_ret_1_dap", "dap"),
    ("abs_ret_1_mop", "mop"),
    ("abs_ret_1_max", "max(urea, dap, mop)"),
]
LAG13_METRICS = [
    ("abs_ret_sum_1_3_urea", "urea"),
    ("abs_ret_sum_1_3_dap", "dap"),
    ("abs_ret_sum_1_3_mop", "mop"),
    ("abs_ret_sum_1_3_max", "max(urea, dap, mop)"),
]


def price_idx_on_or_after(prices: pd.DataFrame, target: pd.Timestamp) -> int | None:
    idx = prices.index[prices["date"] >= target]
    return int(idx[0]) if len(idx) else None


def build_panel(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    daily_stats = news.groupby("news_date")["priority"].agg(["max", "sum", "mean", "count"])
    daily_stats.columns = ["importance_max", "importance_sum", "importance_mean", "news_count"]

    rows = []
    for day, imp_row in daily_stats.iterrows():
        i = price_idx_on_or_after(prices, day)
        if i is None or i + 1 >= len(prices):
            continue

        rec = {
            "news_date": day,
            "price_date_i": prices.loc[i, "date"],
            "importance_max": imp_row["importance_max"],
            "importance_sum": imp_row["importance_sum"],
            "importance_mean": imp_row["importance_mean"],
            "news_count": imp_row["news_count"],
        }

        abs_rets: dict[str, float] = {}
        for product in PRODUCTS:
            y_i = prices.loc[i, PRICE_COLS[product]]
            y_i1 = prices.loc[i + 1, PRICE_COLS[product]]
            abs_rets[product] = abs(y_i1 / y_i - 1)
            rec[f"abs_ret_1_{product}"] = abs_rets[product]

            lag_sum = 0.0
            for k in range(1, 4):
                if i + k < len(prices):
                    y_k = prices.loc[i + k, PRICE_COLS[product]]
                    lag_sum += abs(y_k / y_i - 1)
            rec[f"abs_ret_sum_1_3_{product}"] = lag_sum

        rec["abs_ret_1_max"] = max(abs_rets.values())
        rec["abs_ret_sum_1_3_max"] = max(
            rec[f"abs_ret_sum_1_3_{p}"] for p in PRODUCTS
        )
        rows.append(rec)

    return pd.DataFrame(rows)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), n
    xs, ys = x[mask], y[mask]
    if xs.std() == 0 or ys.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(xs, ys)[0, 1]), n


def spearman(x: pd.Series, y: pd.Series) -> float:
    ranked = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(ranked) < 3:
        return float("nan")
    rx = ranked["x"].rank(method="average")
    ry = ranked["y"].rank(method="average")
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_correlations(
    news_csv: Path | str,
    prices_csv: Path | str = PRICES_PATH,
    importance_col: str = "importance_max",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    news = pd.read_csv(news_csv, parse_dates=["news_date"])
    if start_date is not None:
        news = news[news["news_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        news = news[news["news_date"] <= pd.to_datetime(end_date)]
    prices = (
        pd.read_csv(prices_csv, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    panel = build_panel(news, prices)
    importance = panel[importance_col].values

    lag1: dict[str, dict[str, float | int]] = {}
    lag13: dict[str, dict[str, float | int]] = {}
    spearman_lag1: dict[str, float] = {}
    spearman_lag13: dict[str, float] = {}

    for col, label in LAG1_METRICS:
        r, n = pearson(importance, panel[col].values)
        lag1[label] = {"pearson_r": round(r, 4), "n": n}
        spearman_lag1[label] = round(spearman(panel[importance_col], panel[col]), 4)

    for col, label in LAG13_METRICS:
        r, n = pearson(importance, panel[col].values)
        lag13[label] = {"pearson_r": round(r, 4), "n": n}
        spearman_lag13[label] = round(spearman(panel[importance_col], panel[col]), 4)

    return {
        "news_csv": str(news_csv),
        "prices_csv": str(prices_csv),
        "importance_metric": importance_col,
        "start_date": start_date,
        "end_date": end_date,
        "observations": int(len(panel)),
        "date_range": {
            "from": panel["news_date"].min().strftime("%Y-%m-%d"),
            "to": panel["news_date"].max().strftime("%Y-%m-%d"),
        },
        "lag1_abs_return": lag1,
        "lag1_3_sum_abs_return": lag13,
        "spearman_lag1": spearman_lag1,
        "spearman_lag1_3": spearman_lag13,
    }


def main() -> None:
    news_path = DEFAULT_NEWS_PATH
    result = compute_correlations(news_path)
    df = build_panel(
        pd.read_csv(news_path, parse_dates=["news_date"]),
        pd.read_csv(PRICES_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True),
    )
    importance = df["importance_max"].values

    print(f"Наблюдений (дни с новостями): {result['observations']}")
    print(
        f"Период новостей: {result['date_range']['from']} — {result['date_range']['to']}"
    )
    print()
    print("Метод: Pearson r между max(priority) за день i и метрикой цены")
    print("Цена y_i — первая недельная котировка на дату i или позже; y_{i+k} — k-я следующая неделя")
    print()

    print("1) |y_{i+1}/y_i - 1| (следующая неделя)")
    print("-" * 50)
    for _, label in LAG1_METRICS:
        r = result["lag1_abs_return"][label]["pearson_r"]
        n = result["lag1_abs_return"][label]["n"]
        print(f"  {label:22s}  r = {r:+.4f}  (n={n})")

    print()
    print("2) sum_{k=1..3} |y_{i+k}/y_i - 1| (сумма за 3 следующие недели)")
    print("-" * 50)
    for _, label in LAG13_METRICS:
        r = result["lag1_3_sum_abs_return"][label]["pearson_r"]
        n = result["lag1_3_sum_abs_return"][label]["n"]
        print(f"  {label:22s}  r = {r:+.4f}  (n={n})")

    print()
    print("Spearman rho (importance_max):")
    for label in result["spearman_lag1"]:
        print(f"  {label:22s}  rho = {result['spearman_lag1'][label]:+.4f} (lag1)")
    for label in result["spearman_lag1_3"]:
        print(f"  {label:22s}  rho = {result['spearman_lag1_3'][label]:+.4f} (lag1-3)")


if __name__ == "__main__":
    main()
