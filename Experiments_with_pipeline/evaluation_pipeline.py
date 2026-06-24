from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from solver import KalmanStrategySolver, Solver

ROOT = EXPERIMENT_DIR.parent

PRODUCTS = ("urea", "dap", "mop")
STRATEGY_COLUMNS = {
    "urea": "urea_strategy",
    "dap": "dap_strategy",
    "mop": "mop_strategy",
}
PRICE_COLUMNS = {
    "urea": "urea_price",
    "dap": "dap_price",
    "mop": "mop_price",
}
INITIAL_CAPITAL = 100_000.0
STRATEGY_LABELS = ("buy", "sell", "volatility")


@dataclass
class EvaluationConfig:
    strategy_csv: Path = ROOT / "right_strategy.csv"
    news_csv: Path = ROOT / "market_news.csv"
    start_date: date | None = None
    end_date: date | None = None
    validation: bool = False
    output_csv: Path = Path(__file__).resolve().parent / "evaluation_results.csv"
    summary_json: Path | None = None


def strategy_pnl(strategy: str, return_pct: float) -> float:
    """Доходность позиции: buy — long, sell — short, volatility — без позиции."""
    normalized = strategy.strip().lower()
    if normalized == "buy":
        return return_pct
    if normalized == "sell":
        return -return_pct
    return 0.0


def compound_return_pct(pnl_fractions: pd.Series) -> float:
    """Компаундинг недельных доходностей (в долях), результат в процентах."""
    capital = 1.0
    for value in pnl_fractions:
        if pd.notna(value):
            capital *= 1.0 + float(value)
    return 100.0 * (capital - 1.0)


def summarize_product_pnl(results: pd.DataFrame) -> dict[str, object]:
    """
    Для каждой цены — отдельный компаундинг процентного прироста,
    затем сумма процентных доходностей и прибыли.
    """
    priced = results[results["urea_predicted_pnl"].notna()].copy()
    if priced.empty:
        return {
            "initial_capital": INITIAL_CAPITAL,
            "rows_with_prices": 0,
        }

    by_product_predicted: dict[str, float] = {}
    by_product_benchmark: dict[str, float] = {}
    predicted_profit_by_product: dict[str, float] = {}
    benchmark_profit_by_product: dict[str, float] = {}

    for product in PRODUCTS:
        predicted_col = f"{product}_predicted_pnl"
        actual_col = f"{product}_actual_pnl"

        product_return_pct = compound_return_pct(priced[predicted_col])
        by_product_predicted[product] = round(product_return_pct, 4)
        predicted_profit_by_product[product] = round(
            INITIAL_CAPITAL * product_return_pct / 100.0, 2
        )

        if actual_col in priced.columns and priced[actual_col].notna().any():
            benchmark_return_pct = compound_return_pct(priced[actual_col].dropna())
            by_product_benchmark[product] = round(benchmark_return_pct, 4)
            benchmark_profit_by_product[product] = round(
                INITIAL_CAPITAL * benchmark_return_pct / 100.0, 2
            )

    total_predicted_return_pct = round(sum(by_product_predicted.values()), 4)
    total_benchmark_return_pct = round(sum(by_product_benchmark.values()), 4)
    predicted_profit = round(sum(predicted_profit_by_product.values()), 2)
    benchmark_profit = round(sum(benchmark_profit_by_product.values()), 2)

    return {
        "initial_capital": INITIAL_CAPITAL,
        "rows_with_prices": int(len(priced)),
        "by_product_compound_return_pct": by_product_predicted,
        "by_product_predicted_profit": predicted_profit_by_product,
        "total_sum_compound_return_pct": total_predicted_return_pct,
        "predicted_profit": predicted_profit,
        "predicted_final_capital": round(INITIAL_CAPITAL + predicted_profit, 2),
        "by_product_benchmark_compound_return_pct": by_product_benchmark,
        "by_product_benchmark_profit": benchmark_profit_by_product,
        "benchmark_sum_compound_return_pct": total_benchmark_return_pct,
        "benchmark_profit": benchmark_profit,
        "benchmark_final_capital": round(INITIAL_CAPITAL + benchmark_profit, 2),
    }

def _normalize_strategy_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def build_product_confusion_matrix(
    results: pd.DataFrame,
    product: str,
) -> pd.DataFrame:
    actual_col = f"actual_{product}_strategy"
    predicted_col = f"predicted_{product}_strategy"

    if actual_col not in results.columns or predicted_col not in results.columns:
        raise ValueError(f"Columns for {product} are missing in evaluation results")

    subset = results.copy()
    subset["actual"] = subset[actual_col].map(_normalize_strategy_label)
    subset["predicted"] = subset[predicted_col].map(_normalize_strategy_label)
    subset = subset[subset["actual"].notna() & subset["predicted"].notna()]

    if subset.empty:
        return pd.DataFrame(0, index=STRATEGY_LABELS, columns=STRATEGY_LABELS)

    confusion = pd.crosstab(subset["actual"], subset["predicted"], dropna=False)
    return confusion.reindex(
        index=STRATEGY_LABELS,
        columns=STRATEGY_LABELS,
        fill_value=0,
    )


def plot_confusion_matrix(
    confusion: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_yticks(range(len(confusion.index)))
    ax.set_xticklabels(confusion.columns)
    ax.set_yticklabels(confusion.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = confusion.values.max() / 2 if confusion.values.max() else 0
    for row_idx, actual in enumerate(confusion.index):
        for col_idx, predicted in enumerate(confusion.columns):
            count = int(confusion.loc[actual, predicted])
            color = "white" if count > threshold else "black"
            ax.text(col_idx, row_idx, str(count), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_confusion_matrices_for_evaluation(
    evaluation_csv: Path | str,
    output_dir: Path | str | None = None,
    label: str | None = None,
) -> dict[str, object]:
    evaluation_csv = Path(evaluation_csv)
    if label is None:
        label = evaluation_csv.stem.replace("evaluation_kalman_", "")

    output_dir = Path(output_dir) if output_dir is not None else evaluation_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(evaluation_csv)
    run_summary: dict[str, object] = {
        "label": label,
        "evaluation_csv": str(evaluation_csv),
        "products": {},
    }

    for product in PRODUCTS:
        confusion = build_product_confusion_matrix(results, product)
        csv_path = output_dir / f"confusion_{label}_{product}.csv"
        png_path = output_dir / f"confusion_{label}_{product}.png"
        confusion.to_csv(csv_path, encoding="utf-8-sig")
        plot_confusion_matrix(
            confusion,
            title=f"{label} — {product}",
            output_path=png_path,
        )

        evaluated = int(confusion.values.sum())
        correct = int(sum(confusion.loc[s, s] for s in STRATEGY_LABELS if s in confusion.index))
        run_summary["products"][product] = {
            "evaluated_rows": evaluated,
            "correct": correct,
            "accuracy_pct": round(100.0 * correct / evaluated, 2) if evaluated else None,
            "csv": str(csv_path),
            "png": str(png_path),
            "matrix": {
                actual: {pred: int(confusion.loc[actual, pred]) for pred in STRATEGY_LABELS}
                for actual in STRATEGY_LABELS
            },
        }

    summary_path = output_dir / f"confusion_{label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, ensure_ascii=False, indent=2)

    run_summary["summary_json"] = str(summary_path)
    return run_summary


def build_all_confusion_matrices(
    output_dir: Path | str | None = None,
    evaluation_runs: list[tuple[str, Path]] | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir) if output_dir is not None else EXPERIMENT_DIR
    runs = evaluation_runs or [
        ("market_to_ticker_hybrid", EXPERIMENT_DIR / "evaluation_kalman_market_to_ticker_hybrid.csv"),
        ("market_to_ticker", EXPERIMENT_DIR / "evaluation_kalman_market_to_ticker.csv"),
    ]

    all_summary: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": {},
    }

    for label, evaluation_csv in runs:
        if not evaluation_csv.exists():
            print(f"Skip {label}: file not found ({evaluation_csv})")
            continue
        print(f"Building confusion matrices: {label}")
        all_summary["runs"][label] = build_confusion_matrices_for_evaluation(
            evaluation_csv=evaluation_csv,
            output_dir=output_dir,
            label=label,
        )

    combined_path = output_dir / "confusion_matrices_summary.json"
    with combined_path.open("w", encoding="utf-8") as file:
        json.dump(all_summary, file, ensure_ascii=False, indent=2)

    all_summary["combined_summary_json"] = str(combined_path)
    return all_summary


class EvaluationPipeline:
    def __init__(
        self,
        solver: Solver,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.solver = solver
        self.config = config or EvaluationConfig()

    def _load_strategy(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.strategy_csv, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _load_news(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.news_csv, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _resolve_date_range(
        self, strategy_df: pd.DataFrame, news_df: pd.DataFrame
    ) -> tuple[date, date]:
        min_dates = [strategy_df["date"].min(), news_df["date"].min()]
        max_dates = [strategy_df["date"].max(), news_df["date"].max()]
        start = self.config.start_date or min(min_dates).date()
        end = self.config.end_date or max(max_dates).date()
        return start, end

    @staticmethod
    def _news_for_day(news_df: pd.DataFrame, day: date) -> pd.DataFrame:
        mask = news_df["date"].dt.date == day
        return news_df.loc[mask]

    @staticmethod
    def _news_to_dict(day_news: pd.DataFrame) -> dict[str, str]:
        news: dict[str, str] = {}
        for row in day_news.itertuples(index=False):
            key = f"{row.source}-{row.headline}"
            news[key] = row.headline
        return news

    @staticmethod
    def _strategy_row(strategy_df: pd.DataFrame, day: date) -> pd.Series | None:
        rows = strategy_df[strategy_df["date"].dt.date == day]
        if rows.empty:
            return None
        return rows.iloc[0]

    @staticmethod
    def _has_price(row: pd.Series) -> bool:
        return any(pd.notna(row[PRICE_COLUMNS[product]]) for product in PRODUCTS)

    @staticmethod
    def _compare_strategy(actual: object, predicted: str) -> bool | None:
        if pd.isna(actual):
            return None
        return str(actual).strip().lower() == predicted.strip().lower()

    @staticmethod
    def _returns_at_date(
        strategy_df: pd.DataFrame, day: date
    ) -> dict[str, float] | None:
        dates = strategy_df["date"].dt.date
        mask = dates == day
        if not mask.any():
            return None

        idx = strategy_df.index[mask][0]
        if idx == 0:
            return None

        prev = strategy_df.loc[idx - 1]
        curr = strategy_df.loc[idx]
        returns: dict[str, float] = {}

        for product in PRODUCTS:
            price_col = PRICE_COLUMNS[product]
            if pd.isna(curr[price_col]) or pd.isna(prev[price_col]):
                return None
            returns[product] = float(curr[price_col]) / float(prev[price_col]) - 1.0

        return returns

    def _build_summary(self, results: pd.DataFrame) -> dict[str, object]:
        summary: dict[str, object] = {
            "rows_total": len(results),
            "rows_with_prices": 0,
            "strategy_accuracy": {},
            "pnl": {},
        }

        priced = results[results["urea_predicted_pnl"].notna()]
        summary["rows_with_prices"] = int(len(priced))

        for product in PRODUCTS:
            match_col = f"{product}_strategy_match"
            if match_col in priced.columns:
                valid = priced[match_col].dropna()
                summary["strategy_accuracy"][product] = {
                    "matches": int(valid.sum()),
                    "total": int(len(valid)),
                    "accuracy_pct": round(100.0 * valid.mean(), 2) if len(valid) else None,
                }

        if len(priced) > 0:
            summary["pnl"] = summarize_product_pnl(results)

        return summary

    def _print_summary(self, label: str, summary: dict[str, object]) -> None:
        print(f"\n=== {label} ===")
        print(f"Rows total: {summary['rows_total']}")
        print(f"Rows with price evaluation: {summary['rows_with_prices']}")

        accuracy = summary.get("strategy_accuracy", {})
        for product, stats in accuracy.items():
            if stats["total"]:
                print(
                    f"  {product} accuracy: {stats['matches']}/{stats['total']} "
                    f"({stats['accuracy_pct']}%)"
                )

        pnl = summary.get("pnl", {})
        rows_with_prices = summary.get("rows_with_prices", pnl.get("rows_with_prices", 0))
        if pnl and rows_with_prices > 0:
            print(f"  Total sum compound return: {pnl['total_sum_compound_return_pct']}%")
            for product in PRODUCTS:
                product_pct = pnl["by_product_compound_return_pct"][product]
                product_profit = pnl["by_product_predicted_profit"][product]
                print(
                    f"  {product}: {product_pct}% "
                    f"(profit on {pnl['initial_capital']:.0f}: {product_profit:.2f})"
                )
            print(
                f"  Total predicted profit on {pnl['initial_capital']:.0f}: "
                f"{pnl['predicted_profit']:.2f}"
            )

    def run(self) -> pd.DataFrame:
        strategy_df = self._load_strategy()
        news_df = self._load_news()
        start, end = self._resolve_date_range(strategy_df, news_df)

        rows: list[dict[str, object]] = []
        current = start
        while current <= end:
            day_news = self._news_for_day(news_df, current)
            news_dict = self._news_to_dict(day_news)
            actual_categories = (
                day_news["category"].tolist() if self.config.validation else []
            )

            result = self.solver.get(current.isoformat(), news_dict)
            prediction_date = current + timedelta(days=1)
            target_row = self._strategy_row(strategy_df, prediction_date)
            returns = self._returns_at_date(strategy_df, prediction_date)

            row: dict[str, object] = {
                "news_date": current.isoformat(),
                "prediction_date": prediction_date.isoformat(),
                "news_count": len(result.news),
                "predicted_urea_strategy": result.strategy.get("urea"),
                "predicted_dap_strategy": result.strategy.get("dap"),
                "predicted_mop_strategy": result.strategy.get("mop"),
                "latency_sec": result.latency,
                "token_usage_total": sum(result.token_usage.values()),
                "token_usage": result.token_usage,
            }

            if target_row is not None and self._has_price(target_row):
                for product in PRODUCTS:
                    actual = target_row[STRATEGY_COLUMNS[product]]
                    predicted = result.strategy[product]
                    row[f"actual_{product}_strategy"] = actual
                    row[f"{product}_strategy_match"] = self._compare_strategy(
                        actual, predicted
                    )

            if returns is not None:
                predicted_pnls: list[float] = []
                actual_pnls: list[float] = []
                for product in PRODUCTS:
                    return_pct = returns[product]
                    predicted = result.strategy[product]
                    row[f"{product}_return_pct"] = round(100.0 * return_pct, 4)

                    predicted_pnl = strategy_pnl(predicted, return_pct)
                    row[f"{product}_predicted_pnl"] = predicted_pnl
                    predicted_pnls.append(predicted_pnl)

                    if target_row is not None:
                        actual_strategy = target_row[STRATEGY_COLUMNS[product]]
                        if pd.notna(actual_strategy):
                            actual_pnl = strategy_pnl(str(actual_strategy), return_pct)
                            row[f"{product}_actual_pnl"] = actual_pnl
                            actual_pnls.append(actual_pnl)

                row["total_predicted_pnl"] = sum(predicted_pnls)
                row["sum_predicted_pnl"] = sum(predicted_pnls)
                if actual_pnls:
                    row["total_actual_pnl"] = sum(actual_pnls)
                    row["sum_actual_pnl"] = sum(actual_pnls)

            if self.config.validation:
                matches = [
                    predicted == actual
                    for predicted, actual in zip(result.classes, actual_categories)
                ]
                row["class_matches"] = sum(matches)
                row["class_total"] = len(matches)
                row["predicted_classes"] = result.classes
                row["actual_classes"] = actual_categories

            rows.append(row)
            current += timedelta(days=1)

        results = pd.DataFrame(rows)
        results.to_csv(self.config.output_csv, index=False, encoding="utf-8-sig")

        summary = self._build_summary(results)
        if self.config.summary_json is not None:
            with self.config.summary_json.open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)

        return results, summary


def run_evaluation(
    solver: Solver,
    strategy_csv: Path | str = ROOT / "right_strategy.csv",
    news_csv: Path | str = ROOT / "market_news.csv",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    validation: bool = False,
    output_csv: Path | str | None = None,
    summary_json: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    def _parse_day(value: date | str | None) -> date | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        return datetime.strptime(value, "%Y-%m-%d").date()

    config = EvaluationConfig(
        strategy_csv=Path(strategy_csv),
        news_csv=Path(news_csv),
        start_date=_parse_day(start_date),
        end_date=_parse_day(end_date),
        validation=validation,
        output_csv=Path(output_csv)
        if output_csv is not None
        else EvaluationConfig().output_csv,
        summary_json=Path(summary_json) if summary_json is not None else None,
    )
    pipeline = EvaluationPipeline(solver=solver, config=config)
    return pipeline.run()


def refresh_evaluation_summaries(
    evaluation_csvs: list[Path],
) -> dict[str, dict[str, object]]:
    pipeline = EvaluationPipeline(solver=KalmanStrategySolver())
    refreshed: dict[str, dict[str, object]] = {}

    for evaluation_csv in evaluation_csvs:
        if not evaluation_csv.exists():
            print(f"Skip missing: {evaluation_csv}")
            continue

        results = pd.read_csv(evaluation_csv)
        priced = results[results["urea_predicted_pnl"].notna()]
        summary = {
            "rows_total": len(results),
            "rows_with_prices": int(len(priced)),
            "strategy_accuracy": {},
            "pnl": summarize_product_pnl(results),
        }
        for product in PRODUCTS:
            match_col = f"{product}_strategy_match"
            if match_col in priced.columns:
                valid = priced[match_col].dropna()
                summary["strategy_accuracy"][product] = {
                    "matches": int(valid.sum()),
                    "total": int(len(valid)),
                    "accuracy_pct": round(100.0 * valid.mean(), 2) if len(valid) else None,
                }

        summary_json = evaluation_csv.with_name(
            evaluation_csv.name.replace(".csv", "_summary.json")
        )
        with summary_json.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        label = evaluation_csv.stem.replace("evaluation_kalman_", "")
        refreshed[label] = summary
        pipeline._print_summary(label, summary)
        print(f"  Summary: {summary_json}")

    return refreshed


def run_kalman_evaluations(
    output_dir: Path | str,
    volatility_weight: float = 1.0,
    predictions_runs: list[tuple[str, Path | str]] | None = None,
    build_confusion: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = predictions_runs or [
        ("market_to_ticker_hybrid", ROOT / "Results" / "market_to_ticker_hybrid.csv"),
        ("market_to_ticker", ROOT / "Results" / "market_to_ticker.csv"),
    ]

    pipeline = EvaluationPipeline(solver=KalmanStrategySolver())
    run_summaries: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "volatility_weight": volatility_weight,
        "output_dir": str(output_dir),
        "runs": {},
    }

    for label, predictions_csv in runs:
        solver = KalmanStrategySolver(
            test_mode=True,
            predictions_csv=predictions_csv,
            volatility_weight=volatility_weight,
        )
        output_csv = output_dir / f"evaluation_kalman_{label}.csv"
        summary_json = output_dir / f"evaluation_kalman_{label}_summary.json"

        _, summary = run_evaluation(
            solver=solver,
            output_csv=output_csv,
            summary_json=summary_json,
        )
        summary["volatility_weight"] = volatility_weight
        run_summaries["runs"][label] = summary
        pipeline._print_summary(label, summary)
        print(f"  CSV: {output_csv}")
        print(f"  Summary: {summary_json}")

    if build_confusion:
        confusion_runs = [
            (label, output_dir / f"evaluation_kalman_{label}.csv") for label, _ in runs
        ]
        confusion_summary = build_all_confusion_matrices(
            output_dir=output_dir,
            evaluation_runs=confusion_runs,
        )
        run_summaries["confusion_matrices"] = confusion_summary

    combined_path = output_dir / "evaluation_summary.json"
    with combined_path.open("w", encoding="utf-8") as file:
        json.dump(run_summaries, file, ensure_ascii=False, indent=2)

    run_summaries["combined_summary_json"] = str(combined_path)
    return run_summaries


def main() -> None:
    runs = [
        (
            "market_to_ticker_hybrid",
            ROOT / "Results" / "market_to_ticker_hybrid.csv",
        ),
        (
            "market_to_ticker",
            ROOT / "Results" / "market_to_ticker.csv",
        ),
    ]

    for label, predictions_csv in runs:
        solver = KalmanStrategySolver(
            test_mode=True,
            predictions_csv=predictions_csv,
        )
        output_csv = EXPERIMENT_DIR / f"evaluation_kalman_{label}.csv"
        summary_json = EXPERIMENT_DIR / f"evaluation_kalman_{label}_summary.json"

        _, summary = run_evaluation(
            solver=solver,
            output_csv=output_csv,
            summary_json=summary_json,
        )

        pipeline = EvaluationPipeline(solver=solver)
        pipeline._print_summary(label, summary)
        print(f"  CSV: {output_csv}")
        print(f"  Summary: {summary_json}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "confusion":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else EXPERIMENT_DIR
        summary = build_all_confusion_matrices(output_dir=out)
        for label, run in summary.get("runs", {}).items():
            print(f"\n=== {label} ===")
            for product, stats in run.get("products", {}).items():
                print(
                    f"  {product}: {stats['correct']}/{stats['evaluated_rows']} "
                    f"({stats['accuracy_pct']}%)"
                )
        print(f"\nCombined summary: {summary['combined_summary_json']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "refresh":
        csvs = [
            EXPERIMENT_DIR / "evaluation_kalman_market_to_ticker_hybrid.csv",
            EXPERIMENT_DIR / "evaluation_kalman_market_to_ticker.csv",
            EXPERIMENT_DIR / "results_volatility_weight_0.8" / "evaluation_kalman_market_to_ticker_hybrid.csv",
            EXPERIMENT_DIR / "results_volatility_weight_0.8" / "evaluation_kalman_market_to_ticker.csv",
        ]
        refresh_evaluation_summaries(csvs)
    elif len(sys.argv) > 1 and sys.argv[1] == "vol-weight":
        weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
        folder = (
            Path(sys.argv[3])
            if len(sys.argv) > 3
            else EXPERIMENT_DIR / f"results_vol_weight_{weight}"
        )
        run_kalman_evaluations(output_dir=folder, volatility_weight=weight)
    else:
        main()
