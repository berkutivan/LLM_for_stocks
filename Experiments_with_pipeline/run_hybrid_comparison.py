"""Перегенерация finbert_hybrid, сравнение с baseline и оценка стратегии (gain=0.5)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
RESULTS_DIR = ROOT / "Results"
RESEARCH_DIR = ROOT / "research"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from compare_categories import _load_predicted, compare_categories
from evaluation_pipeline import run_evaluation
from predict_tickers import predict_tickers
from priority_price_correlation import compute_correlations
from solver import KalmanStrategySolver

BASELINE_HYBRID = RESULTS_DIR / "market_to_ticker_hybrid.csv"
GAIN = 0.5


def _log(message: str) -> None:
    print(message, flush=True)


def _compare_new_vs_old(
    new_csv: Path,
    old_csv: Path,
    output_csv: Path,
    output_summary: Path,
) -> dict[str, object]:
    new_df = _load_predicted(new_csv)
    old_df = _load_predicted(old_csv)

    merged = new_df.merge(
        old_df,
        on=["news_date", "source", "headline"],
        how="inner",
        suffixes=("_new", "_old"),
    )
    merged["category_match"] = (
        merged["predicted_category_new"] == merged["predicted_category_old"]
    )
    merged["priority_delta"] = merged["priority_new"] - merged["priority_old"]

    matched = int(merged["category_match"].sum())
    total = len(merged)
    accuracy = matched / total if total else 0.0

    mismatch_pairs = (
        merged.loc[~merged["category_match"]]
        .groupby(["predicted_category_old", "predicted_category_new"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "new_csv": str(new_csv),
        "old_csv": str(old_csv),
        "overlap_rows": total,
        "category_match_rows": matched,
        "category_mismatch_rows": total - matched,
        "category_agreement": round(accuracy, 4),
        "priority_mean_delta_new_minus_old": round(float(merged["priority_delta"].mean()), 4),
        "priority_mae": round(float(merged["priority_delta"].abs().mean()), 4),
        "top_category_mismatches": mismatch_pairs[:20],
    }

    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    _log(
        f"New vs old category agreement: {accuracy:.2%} "
        f"({matched}/{total})"
    )
    return summary


def _run_strategy_eval(
    label: str,
    predictions_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    solver = KalmanStrategySolver(
        gain=GAIN,
        test_mode=True,
        predictions_csv=predictions_csv,
    )
    output_csv = output_dir / f"evaluation_kalman_{label}.csv"
    summary_json = output_dir / f"evaluation_kalman_{label}_summary.json"
    _, summary = run_evaluation(
        solver=solver,
        output_csv=output_csv,
        summary_json=summary_json,
    )
    summary["gain"] = GAIN
    summary["predictions_csv"] = str(predictions_csv)
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def run_comparison(
    output_dir: Path | str | None = None,
    skip_regenerate: bool = False,
    batch_size: int = 10,
) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) if output_dir else RESULTS_DIR / f"hybrid_comparison_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)

    new_csv = out / "market_to_ticker_hybrid_new.csv"
    new_report = out / "market_to_ticker_hybrid_new_report.json"

    if not skip_regenerate:
        if new_csv.exists():
            new_csv.unlink()
        if new_report.exists():
            new_report.unlink()

        _log(f"=== Regenerating finbert_hybrid predictions -> {new_csv} ===")
        predict_tickers(
            news_csv=ROOT / "market_news.csv",
            output_csv=new_csv,
            report_json=new_report,
            batch_size=batch_size,
            analysis_mode="hybrid",
        )
    else:
        _log(f"=== Skipping regeneration, using {new_csv} ===")
        if not new_csv.exists():
            raise FileNotFoundError(f"Expected existing predictions at {new_csv}")

    baseline_copy = out / "market_to_ticker_hybrid_baseline.csv"
    if BASELINE_HYBRID.exists():
        shutil.copy2(BASELINE_HYBRID, baseline_copy)

    _log("")
    _log("=== Category comparison: new vs actual (market_news.csv) ===")
    _, cat_actual_summary = compare_categories(
        predicted_csv=new_csv,
        actual_csv=ROOT / "market_news.csv",
        output_csv=out / "category_comparison_new_vs_actual.csv",
        output_summary=out / "category_comparison_new_vs_actual_summary.json",
    )

    cat_old_summary: dict[str, object] | None = None
    if BASELINE_HYBRID.exists():
        _log("")
        _log("=== Category comparison: new vs baseline hybrid ===")
        cat_old_summary = _compare_new_vs_old(
            new_csv=new_csv,
            old_csv=BASELINE_HYBRID,
            output_csv=out / "category_comparison_new_vs_baseline.csv",
            output_summary=out / "category_comparison_new_vs_baseline_summary.json",
        )

    _log("")
    _log("=== Priority-price correlations ===")
    corr_new = compute_correlations(new_csv)
    corr_baseline = (
        compute_correlations(BASELINE_HYBRID) if BASELINE_HYBRID.exists() else None
    )
    corr_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "new_predictions": corr_new,
        "baseline_predictions": corr_baseline,
    }
    corr_path = out / "priority_price_correlation.json"
    with corr_path.open("w", encoding="utf-8") as file:
        json.dump(corr_payload, file, ensure_ascii=False, indent=2)
    _log(f"Saved correlations: {corr_path}")

    _log("")
    _log(f"=== Strategy evaluation (Kalman gain={GAIN}) ===")
    eval_new = _run_strategy_eval("hybrid_new", new_csv, out)
    eval_baseline: dict[str, object] | None = None
    if BASELINE_HYBRID.exists():
        eval_baseline = _run_strategy_eval("hybrid_baseline", BASELINE_HYBRID, out)

    comparison = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out),
        "gain": GAIN,
        "new_predictions_csv": str(new_csv),
        "baseline_predictions_csv": str(BASELINE_HYBRID),
        "category_vs_actual": cat_actual_summary,
        "category_new_vs_baseline": cat_old_summary,
        "priority_price_correlation": corr_payload,
        "evaluation_new": eval_new,
        "evaluation_baseline": eval_baseline,
    }

    if eval_new and eval_baseline:
        new_pnl = eval_new.get("pnl", {})
        base_pnl = eval_baseline.get("pnl", {})
        comparison["evaluation_delta"] = {
            "total_sum_compound_return_pct": round(
                float(new_pnl.get("total_sum_compound_return_pct", 0))
                - float(base_pnl.get("total_sum_compound_return_pct", 0)),
                4,
            ),
            "predicted_profit": round(
                float(new_pnl.get("predicted_profit", 0))
                - float(base_pnl.get("predicted_profit", 0)),
                2,
            ),
            "by_product_compound_return_pct": {
                product: round(
                    float(new_pnl.get("by_product_compound_return_pct", {}).get(product, 0))
                    - float(base_pnl.get("by_product_compound_return_pct", {}).get(product, 0)),
                    4,
                )
                for product in ("urea", "dap", "mop")
            },
        }

    summary_path = out / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    _log("")
    _log("=== Summary ===")
    _log(f"Output folder: {out}")
    _log(
        f"Category accuracy (new vs actual): "
        f"{cat_actual_summary['accuracy']:.2%}"
    )
    if cat_old_summary:
        _log(
            f"Category agreement (new vs baseline): "
            f"{cat_old_summary['category_agreement']:.2%}"
        )
    if eval_new:
        new_pnl = eval_new.get("pnl", {})
        _log(
            f"Strategy return new: "
            f"{new_pnl.get('total_sum_compound_return_pct')}% "
            f"(profit {new_pnl.get('predicted_profit')})"
        )
    if eval_baseline:
        base_pnl = eval_baseline.get("pnl", {})
        _log(
            f"Strategy return baseline: "
            f"{base_pnl.get('total_sum_compound_return_pct')}% "
            f"(profit {base_pnl.get('predicted_profit')})"
        )
    _log(f"Full summary: {summary_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate finbert_hybrid and compare with baseline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder under Results/ (default: hybrid_comparison_<timestamp>)",
    )
    parser.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Skip LLM regeneration; reuse market_to_ticker_hybrid_new.csv in output dir",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    run_comparison(
        output_dir=args.output_dir,
        skip_regenerate=args.skip_regenerate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
