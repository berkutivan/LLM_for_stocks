from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
RESULTS_DIR = ROOT / "Results"


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class CompareConfig:
    predicted_csv: Path = RESULTS_DIR / "market_to_ticker.csv"
    actual_csv: Path = ROOT / "market_news.csv"
    output_csv: Path = RESULTS_DIR / "category_comparison.csv"
    output_summary: Path = RESULTS_DIR / "category_comparison_summary.json"


def _normalize_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _normalize_text(value: object) -> str:
    return str(value).strip()


def _load_predicted(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"news_date", "source", "headline", "predicted_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["news_date"] = df["news_date"].map(_normalize_date)
    df["source"] = df["source"].map(_normalize_text)
    df["headline"] = df["headline"].map(_normalize_text)
    df["predicted_category"] = df["predicted_category"].map(_normalize_text)
    return df


def _load_actual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "category" not in df.columns and "actual_category" not in df.columns:
        raise ValueError(
            f"{path} has no category labels. "
            "Expected column 'category' (e.g. market_news.csv)."
        )

    date_col = "news_date" if "news_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError(f"{path} has no date column ('date' or 'news_date').")

    category_col = "actual_category" if "actual_category" in df.columns else "category"
    if "source" not in df.columns or "headline" not in df.columns:
        raise ValueError(f"{path} must contain 'source' and 'headline' columns.")

    actual = df[[date_col, "source", "headline", category_col]].copy()
    actual = actual.rename(
        columns={date_col: "news_date", category_col: "actual_category"}
    )
    actual["news_date"] = actual["news_date"].map(_normalize_date)
    actual["source"] = actual["source"].map(_normalize_text)
    actual["headline"] = actual["headline"].map(_normalize_text)
    actual["actual_category"] = actual["actual_category"].map(_normalize_text)
    return actual.drop_duplicates(subset=["news_date", "source", "headline"])


def _per_class_metrics(
    compared: pd.DataFrame,
    classes: list[str],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for category in classes:
        predicted_is = compared["predicted_category"] == category
        actual_is = compared["actual_category"] == category

        tp = int((predicted_is & actual_is).sum())
        fp = int((predicted_is & ~actual_is).sum())
        fn = int((~predicted_is & actual_is).sum())
        support = int(actual_is.sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        metrics[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return metrics


def _build_summary(compared: pd.DataFrame) -> dict[str, object]:
    evaluated = compared[
        compared["predicted_category"].notna() & compared["actual_category"].notna()
    ].copy()

    total = len(compared)
    evaluated_count = len(evaluated)
    matched = int(evaluated["category_match"].sum()) if evaluated_count else 0
    accuracy = matched / evaluated_count if evaluated_count else 0.0

    classes = sorted(
        set(evaluated["actual_category"].tolist())
        | set(evaluated["predicted_category"].tolist())
    )
    per_class = _per_class_metrics(evaluated, classes)

    confusion = pd.crosstab(
        evaluated["actual_category"],
        evaluated["predicted_category"],
        dropna=False,
    )
    confusion_dict = {
        str(actual): {
            str(predicted): int(confusion.loc[actual, predicted])
            for predicted in confusion.columns
        }
        for actual in confusion.index
    }

    mismatches = evaluated.loc[~evaluated["category_match"]]
    mismatch_pairs = (
        mismatches.groupby(["actual_category", "predicted_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    missing_predictions = int(compared["predicted_category"].isna().sum())
    missing_actual = int(compared["actual_category"].isna().sum())
    unmatched_keys = int(
        (compared["predicted_category"].isna() | compared["actual_category"].isna()).sum()
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "predicted_csv": str(compared.attrs.get("predicted_csv", "")),
        "actual_csv": str(compared.attrs.get("actual_csv", "")),
        "total_rows": total,
        "evaluated_rows": evaluated_count,
        "matched_rows": matched,
        "mismatched_rows": evaluated_count - matched,
        "accuracy": round(accuracy, 4),
        "missing_predictions": missing_predictions,
        "missing_actual_labels": missing_actual,
        "rows_with_missing_labels": unmatched_keys,
        "per_class": per_class,
        "confusion_matrix": confusion_dict,
        "top_mismatches": mismatch_pairs[:20],
    }


def compare_categories(
    predicted_csv: Path | str = CompareConfig.predicted_csv,
    actual_csv: Path | str = CompareConfig.actual_csv,
    output_csv: Path | str | None = None,
    output_summary: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = CompareConfig(
        predicted_csv=Path(predicted_csv),
        actual_csv=Path(actual_csv),
        output_csv=Path(output_csv)
        if output_csv is not None
        else CompareConfig().output_csv,
        output_summary=Path(output_summary)
        if output_summary is not None
        else CompareConfig().output_summary,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    predicted = _load_predicted(config.predicted_csv)
    actual = _load_actual(config.actual_csv)

    compared = predicted.merge(
        actual,
        on=["news_date", "source", "headline"],
        how="left",
    )
    compared["category_match"] = (
        compared["predicted_category"] == compared["actual_category"]
    )
    compared.attrs["predicted_csv"] = str(config.predicted_csv)
    compared.attrs["actual_csv"] = str(config.actual_csv)

    summary = _build_summary(compared)

    compared.to_csv(config.output_csv, index=False, encoding="utf-8-sig")
    with config.output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    _log(f"Predicted: {config.predicted_csv}")
    _log(f"Actual:    {config.actual_csv}")
    _log(f"Rows:      {summary['total_rows']} | evaluated: {summary['evaluated_rows']}")
    _log(f"Accuracy:  {summary['accuracy']:.2%} ({summary['matched_rows']}/{summary['evaluated_rows']})")
    _log(f"Saved:     {config.output_csv}")
    _log(f"Saved:     {config.output_summary}")

    return compared, summary


def main() -> None:
    compare_categories()


if __name__ == "__main__":
    main()
