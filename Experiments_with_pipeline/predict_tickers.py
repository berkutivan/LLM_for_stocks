from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
PIPELINE_DIR = ROOT / "Pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import run_ticker_pipeline
from schemas import AnalysisMode, GraphState, Ticker

MAX_ATTEMPTS = 10
BATCH_SIZE = 10
TICKER_DECIMALS = 4

CSV_COLUMNS = [
    "news_date",
    "source",
    "headline",
    "predicted_category",
    "priority",
    "urea_increase",
    "urea_decrease",
    "urea_volatility",
    "dap_increase",
    "dap_decrease",
    "dap_volatility",
    "mop_increase",
    "mop_decrease",
    "mop_volatility",
]


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class PredictConfig:
    news_csv: Path = ROOT / "market_news.csv"
    start_date: date | None = None
    end_date: date | None = None
    output_csv: Path = ROOT / "Results" / "market_to_ticker.csv"
    report_json: Path = ROOT / "Results" / "market_to_ticker_report.json"
    max_attempts: int = MAX_ATTEMPTS
    batch_size: int = BATCH_SIZE
    analysis_mode: AnalysisMode = "standard"


@dataclass(frozen=True)
class NewsItem:
    news_date: str
    source: str
    headline: str
    actual_category: str

    @property
    def news_key(self) -> str:
        return f"{self.source}-{self.headline}"


@dataclass
class ProcessResult:
    csv_row: dict[str, object]
    report_entry: dict[str, object]


def _parse_day(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_news(config: PredictConfig) -> pd.DataFrame:
    df = pd.read_csv(config.news_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if config.start_date is not None:
        df = df[df["date"].dt.date >= config.start_date]
    if config.end_date is not None:
        df = df[df["date"].dt.date <= config.end_date]
    return df.reset_index(drop=True)


def _news_items_from_df(news_df: pd.DataFrame) -> list[NewsItem]:
    return [
        NewsItem(
            news_date=row.date.strftime("%Y-%m-%d"),
            source=row.source,
            headline=row.headline,
            actual_category=row.category,
        )
        for row in news_df.itertuples(index=False)
    ]


def _is_pipeline_success(state: GraphState) -> bool:
    if state.error_messages:
        return False
    news_count = len(state.news)
    if news_count == 0:
        return True
    return (
        len(state.classified_news) == news_count
        and len(state.prioritased_news) == news_count
        and len(state.ticker_from_news) == news_count
    )


def _round_prob(value: float) -> float:
    return round(value, TICKER_DECIMALS)


def _ticker_csv_fields(prefix: str, ticker: Ticker) -> dict[str, float]:
    return {
        f"{prefix}_increase": _round_prob(ticker.Increase_price),
        f"{prefix}_decrease": _round_prob(ticker.Decrease_price),
        f"{prefix}_volatility": _round_prob(ticker.Volatility_price),
    }


def _empty_ticker_fields() -> dict[str, None]:
    return {
        f"{product}_{component}": None
        for product in ("urea", "dap", "mop")
        for component in ("increase", "decrease", "volatility")
    }


def _report_entry(
    item: NewsItem,
    state: GraphState | None,
    attempts: int,
    run_error: str | None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "news_date": item.news_date,
        "news_key": item.news_key,
        "source": item.source,
        "headline": item.headline,
        "actual_category": item.actual_category,
        "attempts": attempts,
        "run_error": run_error,
    }
    if state is None:
        entry.update(
            {
                "token_use": {},
                "token_use_total": 0,
                "time_use_sec": {},
                "pipeline_errors": [],
            }
        )
        return entry

    entry.update(
        {
            "token_use": dict(state.token_use),
            "token_use_total": sum(state.token_use.values()),
            "time_use_sec": {
                key: round(value, 6) for key, value in state.time_use.items()
            },
            "pipeline_errors": list(state.error_messages),
        }
    )
    return entry


def _result_from_state(
    item: NewsItem,
    state: GraphState | None,
    attempts: int,
    run_error: str | None,
) -> ProcessResult:
    csv_row: dict[str, object] = {
        "news_date": item.news_date,
        "source": item.source,
        "headline": item.headline,
        "predicted_category": None,
        "priority": None,
    }
    csv_row.update(_empty_ticker_fields())

    if state is not None and state.ticker_from_news:
        ticker = state.ticker_from_news[0]
        csv_row["predicted_category"] = (
            state.classified_news[0] if state.classified_news else None
        )
        csv_row["priority"] = (
            state.prioritased_news[0] if state.prioritased_news else None
        )
        csv_row.update(_ticker_csv_fields("urea", ticker.urea_ticker))
        csv_row.update(_ticker_csv_fields("dap", ticker.dap_ticker))
        csv_row.update(_ticker_csv_fields("mop", ticker.mop_ticker))

    return ProcessResult(
        csv_row=csv_row,
        report_entry=_report_entry(item, state, attempts, run_error),
    )


def _process_single_news(
    item: NewsItem,
    max_attempts: int,
    progress_prefix: str,
    analysis_mode: AnalysisMode = "standard",
) -> ProcessResult:
    """Обрабатывает одну новость; повторяет до успеха (остальные в батче ждут)."""
    news = {item.news_key: item.headline}
    attempt = 0
    last_error: str | None = None
    last_state: GraphState | None = None

    while True:
        attempt += 1
        if attempt > 1:
            _log(f"{progress_prefix}  retry {attempt}...")

        try:
            state = run_ticker_pipeline(
                data=item.news_date,
                news=news,
                analysis_mode=analysis_mode,
            )
        except Exception as exc:
            last_error = str(exc)
            last_state = None
            _log(f"{progress_prefix}  attempt {attempt} failed: {exc}")
        else:
            last_state = state
            if _is_pipeline_success(state):
                if attempt > 1:
                    _log(f"{progress_prefix}  OK on attempt {attempt}")
                return _result_from_state(item, state, attempt, None)

            last_error = (
                "; ".join(state.error_messages)
                if state.error_messages
                else "incomplete pipeline response"
            )
            _log(f"{progress_prefix}  attempt {attempt} incomplete: {last_error}")

        if attempt >= max_attempts and attempt % max_attempts == 0:
            _log(
                f"{progress_prefix}  {max_attempts} attempts reached, "
                "continuing retries until success..."
            )


def _append_batch_to_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    df.to_csv(
        output_csv,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
        float_format=f"%.{TICKER_DECIMALS}f",
    )


def _load_report(report_json: Path) -> dict[str, object]:
    if not report_json.exists() or report_json.stat().st_size == 0:
        return {"items": []}
    with report_json.open(encoding="utf-8") as file:
        return json.load(file)


def _append_batch_to_report(
    report_json: Path,
    batch_index: int,
    entries: list[dict[str, object]],
    batch_elapsed: float,
) -> None:
    report = _load_report(report_json)
    items = report.setdefault("items", [])
    if not isinstance(items, list):
        items = []
        report["items"] = items

    items.extend(entries)
    batches = report.setdefault("batches", [])
    if not isinstance(batches, list):
        batches = []
        report["batches"] = batches

    batches.append(
        {
            "batch_index": batch_index,
            "rows": len(entries),
            "elapsed_sec": round(batch_elapsed, 3),
            "token_use_total": sum(
                int(entry.get("token_use_total", 0)) for entry in entries
            ),
        }
    )
    report["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report["total_items"] = len(items)

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def _process_batch(
    batch_index: int,
    total_batches: int,
    batch: list[NewsItem],
    config: PredictConfig,
) -> tuple[list[ProcessResult], int, float]:
    batch_prefix = f"[batch {batch_index}/{total_batches}]"
    _log(f"{batch_prefix} start | news: {len(batch)} | parallel workers: {len(batch)}")

    batch_started = perf_counter()
    workers = min(len(batch), config.batch_size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda item: _process_single_news(
                    item,
                    config.max_attempts,
                    f"{batch_prefix} {item.news_key[:60]}",
                    config.analysis_mode,
                ),
                batch,
            )
        )

    batch_tokens = sum(result.report_entry.get("token_use_total", 0) for result in results)
    batch_elapsed = perf_counter() - batch_started
    _log(
        f"{batch_prefix} done | rows: {len(results)} | "
        f"tokens: {batch_tokens} | {batch_elapsed:.1f}s"
    )
    return results, batch_tokens, batch_elapsed


def predict_tickers(
    news_csv: Path | str = ROOT / "market_news.csv",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    output_csv: Path | str | None = None,
    report_json: Path | str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    batch_size: int = BATCH_SIZE,
    analysis_mode: AnalysisMode = "standard",
) -> pd.DataFrame:
    config = PredictConfig(
        news_csv=Path(news_csv),
        start_date=_parse_day(start_date),
        end_date=_parse_day(end_date),
        output_csv=Path(output_csv)
        if output_csv is not None
        else PredictConfig().output_csv,
        report_json=Path(report_json)
        if report_json is not None
        else PredictConfig().report_json,
        max_attempts=max_attempts,
        batch_size=batch_size,
        analysis_mode=analysis_mode,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    news_df = _load_news(config)
    if news_df.empty:
        _log("No news rows matched the filter; nothing to predict.")
        empty = pd.DataFrame(columns=CSV_COLUMNS)
        empty.to_csv(config.output_csv, index=False, encoding="utf-8-sig")
        with config.report_json.open("w", encoding="utf-8") as file:
            json.dump({"items": [], "batches": []}, file, ensure_ascii=False, indent=2)
        return empty

    items = _news_items_from_df(news_df)
    total_news = len(items)
    total_batches = (total_news + config.batch_size - 1) // config.batch_size
    all_rows: list[dict[str, object]] = []
    total_tokens = 0
    started = perf_counter()

    _log(f"Input: {config.news_csv}")
    _log(f"Analysis mode: {config.analysis_mode}")
    _log(f"Output CSV: {config.output_csv}")
    _log(f"Output report: {config.report_json}")
    _log(
        f"News items: {total_news} | batches: {total_batches} | "
        f"batch size: {config.batch_size}"
    )
    _log("")

    for batch_index, start in enumerate(range(0, total_news, config.batch_size), start=1):
        batch = items[start : start + config.batch_size]
        batch_results, batch_tokens, batch_elapsed = _process_batch(
            batch_index=batch_index,
            total_batches=total_batches,
            batch=batch,
            config=config,
        )
        csv_rows = [result.csv_row for result in batch_results]
        report_entries = [result.report_entry for result in batch_results]

        _append_batch_to_csv(config.output_csv, csv_rows)
        _append_batch_to_report(
            config.report_json,
            batch_index=batch_index,
            entries=report_entries,
            batch_elapsed=batch_elapsed,
        )

        all_rows.extend(csv_rows)
        total_tokens += batch_tokens
        _log(
            f"[batch {batch_index}/{total_batches}] saved | "
            f"csv: {config.output_csv.name} | report: {config.report_json.name}"
        )
        _log("")

    results = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    elapsed = perf_counter() - started
    _log(
        f"Done: {len(results)} rows | csv: {config.output_csv} | "
        f"report: {config.report_json} | batches: {total_batches} | "
        f"tokens total: {total_tokens} | elapsed: {elapsed:.1f}s"
    )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Batch ticker predictions via LLM pipeline")
    parser.add_argument(
        "--analysis-mode",
        choices=("standard", "hybrid"),
        default="standard",
        help="Pipeline mode: standard (LLM only) or hybrid (FinBERT + LLM)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default depends on analysis mode)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Output report JSON path",
    )
    parser.add_argument("--news-csv", type=Path, default=ROOT / "market_news.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    default_csv = (
        ROOT / "Results" / "market_to_ticker_hybrid.csv"
        if args.analysis_mode == "hybrid"
        else ROOT / "Results" / "market_to_ticker.csv"
    )
    default_report = default_csv.with_name(
        default_csv.stem + "_report.json"
    )

    predict_tickers(
        news_csv=args.news_csv,
        output_csv=args.output_csv or default_csv,
        report_json=args.report_json or default_report,
        batch_size=args.batch_size,
        analysis_mode=args.analysis_mode,
    )


if __name__ == "__main__":
    main()
