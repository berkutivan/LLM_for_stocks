"""Run pipeline on one day from market_news.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from llm.llm_schemas import get_model_name
from pipeline import run_ticker_pipeline

NEWS_CSV = ROOT / "market_news.csv"
EXAMPLE_DATE = "2020-01-10"


def news_for_day(news_csv: Path, day: str) -> dict[str, str]:
    df = pd.read_csv(news_csv, parse_dates=["date"])
    mask = df["date"].dt.strftime("%Y-%m-%d") == day
    day_news = df.loc[mask]
    return {
        f"{row.source}-{row.headline}": row.headline
        for row in day_news.itertuples(index=False)
    }


def main() -> None:
    news = news_for_day(NEWS_CSV, EXAMPLE_DATE)
    print(f"Date: {EXAMPLE_DATE}")
    print(f"News count: {len(news)}")
    print(
        "Models:",
        json.dumps(
            {
                "prioritizer": get_model_name("prioritizer"),
                "ticker": get_model_name("ticker"),
                "helper": get_model_name("helper"),
            },
            ensure_ascii=False,
        ),
    )
    print()

    for key, headline in news.items():
        print(f"  - {key}: {headline}")
    print()

    state = run_ticker_pipeline(data=EXAMPLE_DATE, news=news)

    for index, key in enumerate(state.news):
        print(f"[{index}] {key}")
        print(f"    class:    {state.classified_news[index]}")
        print(f"    priority: {state.prioritased_news[index]}")
        ticker = state.ticker_from_news[index]
        print(
            "    urea:",
            ticker.urea_ticker.model_dump(),
        )
        print(
            "    dap:",
            ticker.dap_ticker.model_dump(),
        )
        print(
            "    mop:",
            ticker.mop_ticker.model_dump(),
        )
        print()

    print("token_use:", state.token_use)
    print("time_use:", {k: round(v, 3) for k, v in state.time_use.items()})
    if state.error_messages:
        print("errors:", state.error_messages)


if __name__ == "__main__":
    main()
