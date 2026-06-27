from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.split import temporal_split
from src.prices.streaks import compute_streaks_for_news_df


def fact_cache_key(source: str, headline: str) -> str:
    return f"{source}|||{headline}"


def deduplicate_streak_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["source", "headline"], as_index=False).agg(
        {"date": "min", "streaks": "first", "fact_text": "first", "index_text": "first"}
    )
    return grouped.rename(columns={"date": "date_first_seen"})


def build_facts_corpus(
    config: dict[str, Any],
    news_path: Path,
    prices_path: Path,
    output_path: Path,
    quiet: bool = False,
) -> int:
    news = pd.read_csv(news_path, parse_dates=["date"]).sort_values("date")
    train, _ = temporal_split(news, config["eval"]["train_ratio"])

    streak_cfg = config.get("streak", {})
    max_days = streak_cfg.get("max_days", 4)
    flat_threshold = streak_cfg.get("flat_threshold", 0.01)

    computed = compute_streaks_for_news_df(
        train,
        str(prices_path),
        flat_threshold=flat_threshold,
        mode="days",
        max_days=max_days,
    )
    deduped = deduplicate_streak_rows(computed)

    if not quiet:
        print(f"Train news: {len(train)}, unique (source, headline): {len(deduped)}")

    facts: list[dict] = []
    for _, row in deduped.iterrows():
        facts.append(
            {
                "source": row["source"],
                "headline": row["headline"],
                "date_first_seen": pd.Timestamp(row["date_first_seen"]).strftime("%Y-%m-%d"),
                "streaks": row["streaks"],
                "fact_text": row["fact_text"],
                "index_text": row["index_text"],
            }
        )

    facts.sort(key=lambda f: f["date_first_seen"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in facts:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(facts)


def cmd_build_rag_from_config(
    config_path: str | None = None,
    no_cache: bool = False,
    quiet: bool = False,
) -> None:
    del no_cache
    config = load_config(config_path)
    facts_path = resolve_path(config, config["rag"]["facts_path"])

    count = build_facts_corpus(
        config=config,
        news_path=PROJECT_ROOT / "market_news.csv",
        prices_path=PROJECT_ROOT / "new_prices.csv",
        output_path=facts_path,
        quiet=quiet,
    )

    from src.rag.retriever import HistoricalFactRetriever

    retriever = HistoricalFactRetriever(
        facts_path=facts_path,
        index_dir=resolve_path(config, config["rag"]["index_dir"]),
        embedding_model=config["rag"]["embedding_model"],
        top_k=config["rag"]["top_k"],
        retrieve_candidates=config["rag"].get("retrieve_candidates", 20),
    )
    retriever.initialize(force_rebuild=True)
    print(f"Built {count} streak facts -> {facts_path}")
