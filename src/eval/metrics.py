from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PRODUCTS = ["urea", "dap", "mop"]


def score_importance_from_scores(row: pd.Series) -> float:
    return max(abs(row["score_urea"]), abs(row["score_dap"]), abs(row["score_mop"]))


def pearson_safe(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 3:
        return None
    if x.std() == 0 or y.std() == 0:
        return None
    return float(x.corr(y))


def compute_metrics(
    predictions: pd.DataFrame,
    aligned: pd.DataFrame,
    split: str = "all",
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    merged = predictions.merge(
        aligned,
        on=["date", "source", "headline"],
        how="inner",
        suffixes=("", "_true"),
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    if split == "train":
        cutoff = int(len(merged) * train_ratio)
        merged = merged.iloc[:cutoff]
    elif split == "test":
        cutoff = int(len(merged) * train_ratio)
        merged = merged.iloc[cutoff:]

    if merged.empty:
        return {"error": f"No rows for split={split}", "n": 0}

    merged["importance_scores"] = merged.apply(score_importance_from_scores, axis=1)

    y_true = merged["true_category"]
    y_pred = merged["pred_category"]
    accuracy = float(accuracy_score(y_true, y_pred))

    corr_scores: dict[str, float | None] = {}
    for product in PRODUCTS:
        corr_scores[product] = pearson_safe(
            merged["importance_scores"],
            merged[f"abs_delta_{product}"],
        )
    corr_llm: dict[str, float | None] = {}
    for product in PRODUCTS:
        corr_llm[product] = pearson_safe(
            merged["importance"],
            merged[f"abs_delta_{product}"],
        )

    valid_r = [v for v in corr_scores.values() if v is not None]
    r_mean = float(np.mean(valid_r)) if valid_r else None

    valid_r_llm = [v for v in corr_llm.values() if v is not None]
    r_mean_llm = float(np.mean(valid_r_llm)) if valid_r_llm else None

    signed_corr: dict[str, float | None] = {}
    for product in PRODUCTS:
        signed_corr[product] = pearson_safe(
            merged[f"score_{product}"],
            merged[f"delta_pct_{product}"],
        )

    labels = sorted(set(y_true) | set(y_pred))
    return {
        "split": split,
        "n": len(merged),
        "accuracy": accuracy,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "correlation_importance_from_scores": corr_scores,
        "correlation_importance_from_llm": corr_llm,
        "r_mean": r_mean,
        "r_mean_llm": r_mean_llm,
        "signed_correlation": signed_corr,
    }
