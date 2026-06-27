from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prices.align import PRODUCTS


def _direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true == y_pred).mean())


def compute_next_day_metrics(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    merged = predictions.merge(
        ground_truth,
        on=["date", "source", "headline"],
        how="inner",
    )
    if merged.empty:
        return {"n": 0, "error": "no merged rows"}

    variants = sorted(merged["variant"].unique()) if "variant" in merged.columns else [1]
    report: dict[str, Any] = {"n": len(merged), "by_variant": {}}

    for variant in variants:
        sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
        vreport: dict[str, Any] = {"n": len(sub), "products": {}}
        dir_accs: list[float] = []

        for product in PRODUCTS:
            true_dir = sub[f"true_dir_{product}"]
            pred_dir = sub[f"pred_dir_{product}"]
            dir_acc = _direction_accuracy(true_dir, pred_dir)
            dir_accs.append(dir_acc)
            vreport["products"][product] = {"direction_accuracy": dir_acc}

        vreport["direction_accuracy_mean"] = float(np.mean(dir_accs))
        report["by_variant"][str(variant)] = vreport

    return report


def save_next_day_comparison(report: dict[str, Any], output_path: Path | str) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, vrep in report.get("by_variant", {}).items():
        rows.append(
            {
                "variant": int(variant),
                "n": vrep["n"],
                "direction_accuracy_mean": vrep["direction_accuracy_mean"],
                **{
                    f"{product}_dir_acc": vrep["products"][product]["direction_accuracy"]
                    for product in PRODUCTS
                },
            }
        )
    df = pd.DataFrame(rows).sort_values("variant")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
