from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "Pipeline"
NEWS_CSV = ROOT / "market_news.csv"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def has_llm_credentials() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )


def load_news_for_day(day: str) -> dict[str, str]:
    df = pd.read_csv(NEWS_CSV, parse_dates=["date"])
    mask = df["date"].dt.strftime("%Y-%m-%d") == day
    day_news = df.loc[mask]
    if day_news.empty:
        raise ValueError(f"No news found in {NEWS_CSV} for date {day}")
    return {
        f"{row.source}-{row.headline}": row.headline
        for row in day_news.itertuples(index=False)
    }


@pytest.fixture(scope="session")
def news_csv_path() -> Path:
    assert NEWS_CSV.is_file(), f"Missing dataset: {NEWS_CSV}"
    return NEWS_CSV


@pytest.fixture
def single_news_day() -> tuple[str, dict[str, str]]:
    day = "2020-01-10"
    return day, load_news_for_day(day)


@pytest.fixture
def multi_news_day() -> tuple[str, dict[str, str]]:
    day = "2021-05-24"
    return day, load_news_for_day(day)


@pytest.fixture
def has_openai_key() -> bool:
    return has_llm_credentials()
