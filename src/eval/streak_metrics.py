from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.prices.align import PRODUCTS


def _direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true == y_pred).mean())


def compute_streak_metrics(
    merged: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
) -> dict[str, Any]:
    del ground_truth
    if merged.empty:
        return {"n": 0, "error": "no merged rows"}

    variants = sorted(merged["variant"].unique()) if "variant" in merged.columns else [1]
    report: dict[str, Any] = {"n": len(merged), "by_variant": {}}

    for variant in variants:
        sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
        vreport: dict[str, Any] = {"n": len(sub), "products": {}}
        dir_accs = []
        week_maes = []

        for product in PRODUCTS:
            true_dir = sub[f"true_dir_{product}"]
            pred_dir = sub[f"pred_dir_{product}"]
            true_weeks = sub[f"true_weeks_{product}"].astype(float)
            pred_weeks = sub[f"pred_weeks_{product}"].astype(float)

            dir_acc = _direction_accuracy(true_dir, pred_dir)
            week_mae = float(np.mean(np.abs(true_weeks - pred_weeks)))
            dir_accs.append(dir_acc)
            week_maes.append(week_mae)
            vreport["products"][product] = {
                "direction_accuracy": dir_acc,
                "weeks_mae": week_mae,
            }

        vreport["direction_accuracy_mean"] = float(np.mean(dir_accs))
        vreport["weeks_mae_mean"] = float(np.mean(week_maes))
        report["by_variant"][str(variant)] = vreport

    return report
