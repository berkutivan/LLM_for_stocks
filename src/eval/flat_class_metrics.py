from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prices.align import PRODUCTS
from src.prices.next_day import compute_next_day_returns, directions_from_returns


def flat_precision_recall(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | None]:
    true_flat = y_true == "flat"
    pred_flat = y_pred == "flat"
    tp = int((true_flat & pred_flat).sum())
    fp = int((~true_flat & pred_flat).sum())
    fn = int((true_flat & ~pred_flat).sum())

    precision = float(tp / (tp + fp)) if tp + fp > 0 else None
    recall = float(tp / (tp + fn)) if tp + fn > 0 else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_flat": precision,
        "recall_flat": recall,
        "n_true_flat": int(true_flat.sum()),
        "n_pred_flat": int(pred_flat.sum()),
    }


def sweep_flat_metrics(
    predictions: pd.DataFrame,
    returns_df: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    variants = sorted(predictions["variant"].unique())

    for threshold in thresholds:
        truth = directions_from_returns(returns_df, float(threshold))
        for variant in variants:
            preds = predictions[predictions["variant"] == variant]
            merged = preds.merge(
                truth,
                on=["date", "source", "headline"],
                how="inner",
            )
            for product in PRODUCTS:
                metrics = flat_precision_recall(
                    merged[f"true_dir_{product}"],
                    merged[f"pred_dir_{product}"],
                )
                rows.append(
                    {
                        "flat_threshold": float(threshold),
                        "flat_threshold_pct": float(threshold * 100),
                        "variant": int(variant),
                        "product": product,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def plot_flat_metrics(
    sweep_df: pd.DataFrame,
    output_path: Path | str,
    current_threshold_pct: float | None = None,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    variant_styles = {1: ("v1 RAG", "#1f77b4"), 2: ("v2 +цены", "#ff7f0e"), 3: ("v3 +новости", "#2ca02c")}

    for row_idx, product in enumerate(PRODUCTS):
        ax_p = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        sub_all = sweep_df[sweep_df["product"] == product]

        for variant, (label, color) in variant_styles.items():
            sub = sub_all[sub_all["variant"] == variant].sort_values("flat_threshold_pct")
            ax_p.plot(
                sub["flat_threshold_pct"],
                sub["precision_flat"],
                marker="o",
                ms=3,
                label=label,
                color=color,
            )
            ax_r.plot(
                sub["flat_threshold_pct"],
                sub["recall_flat"],
                marker="o",
                ms=3,
                label=label,
                color=color,
            )

        ax_p.set_ylabel(product.upper())
        ax_p.set_ylim(0, 1.05)
        ax_r.set_ylim(0, 1.05)
        ax_p.grid(True, alpha=0.3)
        ax_r.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_p.set_title("Precision (flat)")
            ax_r.set_title("Recall (flat)")
        if row_idx == 2:
            ax_p.set_xlabel("flat threshold, %")
            ax_r.set_xlabel("flat threshold, %")

    if current_threshold_pct is not None:
        for ax in axes.flat:
            ax.axvline(current_threshold_pct, color="gray", ls="--", lw=1, alpha=0.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Precision / Recall класса flat vs порог flat", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_at_threshold(sweep_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sub = sweep_df[np.isclose(sweep_df["flat_threshold"], threshold)]
    return sub.sort_values(["variant", "product"])


def run_flat_analysis(
    predictions: pd.DataFrame,
    returns_df: pd.DataFrame,
    output_dir: Path,
    threshold_min: float = 0.001,
    threshold_max: float = 0.05,
    threshold_steps: int = 40,
    current_threshold: float = 0.01,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    if not np.any(np.isclose(thresholds, current_threshold)):
        thresholds = np.sort(np.append(thresholds, current_threshold))
    sweep_df = sweep_flat_metrics(predictions, returns_df, thresholds)

    csv_path = output_dir / "flat_precision_recall_sweep.csv"
    sweep_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = output_dir / "flat_precision_recall_plot.png"
    plot_flat_metrics(sweep_df, plot_path, current_threshold_pct=current_threshold * 100)

    at_current = metrics_at_threshold(sweep_df, current_threshold)
    current_path = output_dir / "flat_precision_recall_at_threshold.csv"
    at_current.to_csv(current_path, index=False, encoding="utf-8-sig")

    return {
        "sweep_csv": str(csv_path),
        "plot": str(plot_path),
        "at_threshold_csv": str(current_path),
        "current_threshold": current_threshold,
        "at_current": at_current,
    }
