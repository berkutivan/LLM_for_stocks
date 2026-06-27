from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prices.align import PRODUCTS
from src.prices.streaks import compute_day_streaks_for_news_df


def _days_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true.astype(int) == y_pred.astype(int)).mean())


def _class_distribution(y: pd.Series) -> dict[str, int]:
    counts = y.astype(int).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def normalize_pred_days_value(days: int, weeks: int = 0, direction: str = "up") -> int:
    """Map pred_days to GT class 0–4 (legacy CSV may store weeks×7)."""
    if direction == "flat":
        return 0
    d = int(days)
    w = int(weeks or 0)
    if d > 4:
        if w > 0:
            return min(4, max(1, w))
        if d % 7 == 0:
            return min(4, max(1, d // 7))
        return min(4, d)
    return max(0, d)


def _normalized_pred_days(sub: pd.DataFrame, product: str) -> pd.Series:
    days = sub[f"pred_days_{product}"].astype(int)
    weeks_col = f"pred_weeks_{product}"
    dir_col = f"pred_dir_{product}"
    weeks = sub[weeks_col].astype(int) if weeks_col in sub.columns else pd.Series(0, index=sub.index)
    dirs = sub[dir_col] if dir_col in sub.columns else pd.Series("up", index=sub.index)
    return pd.Series(
        [
            normalize_pred_days_value(d, w, dr)
            for d, w, dr in zip(days, weeks, dirs)
        ],
        index=sub.index,
    )


def _multiclass_precision(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: int | str,
) -> tuple[float | None, int, int]:
    pred_mask = y_pred == label
    n_pred = int(pred_mask.sum())
    if n_pred == 0:
        return None, 0, 0
    tp = int((y_true[pred_mask] == label).sum())
    return float(tp / n_pred), n_pred, tp


def _multiclass_recall(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: int | str,
) -> tuple[float | None, int, int]:
    true_mask = y_true == label
    n_true = int(true_mask.sum())
    if n_true == 0:
        return None, 0, 0
    tp = int((y_pred[true_mask] == label).sum())
    return float(tp / n_true), n_true, tp


def _direction_precision(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str,
) -> tuple[float | None, int, int]:
    return _multiclass_precision(y_true, y_pred, label)


def _direction_recall(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str,
) -> tuple[float | None, int, int]:
    return _multiclass_recall(y_true, y_pred, label)


def _per_day_class_direction_precision(merged: pd.DataFrame, variant: int = 3) -> pd.DataFrame:
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    rows: list[dict] = []
    for product in PRODUCTS:
        true_days = sub[f"true_days_{product}"].astype(int)
        true_dir = sub[f"true_dir_{product}"]
        pred_dir = sub[f"pred_dir_{product}"]
        for day_class in range(5):
            mask = true_days == day_class
            n = int(mask.sum())
            if n == 0:
                rows.append(
                    {
                        "product": product,
                        "true_days": day_class,
                        "n": 0,
                        "precision_flat": None,
                        "n_pred_flat": 0,
                        "tp_flat": 0,
                        "precision_up": None,
                        "n_pred_up": 0,
                        "tp_up": 0,
                        "precision_down": None,
                        "n_pred_down": 0,
                        "tp_down": 0,
                        "recall_flat": None,
                        "n_true_flat": 0,
                        "recall_up": None,
                        "n_true_up": 0,
                        "recall_down": None,
                        "n_true_down": 0,
                    }
                )
                continue
            yt = true_dir[mask]
            yp = pred_dir[mask]
            p_flat, n_pred_flat, tp_flat = _direction_precision(yt, yp, "flat")
            p_up, n_pred_up, tp_up = _direction_precision(yt, yp, "up")
            p_down, n_pred_down, tp_down = _direction_precision(yt, yp, "down")
            r_flat, n_true_flat, _ = _direction_recall(yt, yp, "flat")
            r_up, n_true_up, _ = _direction_recall(yt, yp, "up")
            r_down, n_true_down, _ = _direction_recall(yt, yp, "down")
            rows.append(
                {
                    "product": product,
                    "true_days": day_class,
                    "n": n,
                    "precision_flat": p_flat,
                    "n_pred_flat": n_pred_flat,
                    "tp_flat": tp_flat,
                    "precision_up": p_up,
                    "n_pred_up": n_pred_up,
                    "tp_up": tp_up,
                    "precision_down": p_down,
                    "n_pred_down": n_pred_down,
                    "tp_down": tp_down,
                    "recall_flat": r_flat,
                    "n_true_flat": n_true_flat,
                    "recall_up": r_up,
                    "n_true_up": n_true_up,
                    "recall_down": r_down,
                    "n_true_down": n_true_down,
                }
            )
    return pd.DataFrame(rows)


def _per_day_class_direction_accuracy(merged: pd.DataFrame, variant: int = 3) -> pd.DataFrame:
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    rows: list[dict] = []
    for product in PRODUCTS:
        true_days = sub[f"true_days_{product}"].astype(int)
        true_dir = sub[f"true_dir_{product}"]
        pred_dir = sub[f"pred_dir_{product}"]
        for day_class in range(5):
            mask = true_days == day_class
            n = int(mask.sum())
            if n == 0:
                acc = None
                true_dist: dict[str, int] = {}
                pred_dist: dict[str, int] = {}
            else:
                acc = float((true_dir[mask] == pred_dir[mask]).mean())
                true_dist = true_dir[mask].value_counts().to_dict()
                pred_dist = pred_dir[mask].value_counts().to_dict()
            rows.append(
                {
                    "product": product,
                    "true_days": day_class,
                    "direction_accuracy": acc,
                    "n": n,
                    "true_up": int(true_dist.get("up", 0)),
                    "true_flat": int(true_dist.get("flat", 0)),
                    "true_down": int(true_dist.get("down", 0)),
                    "pred_up": int(pred_dist.get("up", 0)),
                    "pred_flat": int(pred_dist.get("flat", 0)),
                    "pred_down": int(pred_dist.get("down", 0)),
                }
            )
    return pd.DataFrame(rows)


def _per_day_class_accuracy(merged: pd.DataFrame, variant: int = 3) -> pd.DataFrame:
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    rows: list[dict] = []
    for product in PRODUCTS:
        true_days = sub[f"true_days_{product}"].astype(int)
        pred_days = _normalized_pred_days(sub, product)
        for day_class in range(5):
            mask = true_days == day_class
            n = int(mask.sum())
            acc = float((true_days[mask] == pred_days[mask]).mean()) if n > 0 else None
            rows.append(
                {
                    "product": product,
                    "true_days": day_class,
                    "accuracy": acc,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _per_days_count_precision_recall(merged: pd.DataFrame, variant: int = 3) -> pd.DataFrame:
    """Precision/recall for multi-class days prediction (classes 0–4)."""
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    rows: list[dict] = []
    for product in PRODUCTS:
        true_days = sub[f"true_days_{product}"].astype(int)
        pred_days = _normalized_pred_days(sub, product)
        for day_class in range(5):
            p, n_pred, tp_p = _multiclass_precision(true_days, pred_days, day_class)
            r, n_true, tp_r = _multiclass_recall(true_days, pred_days, day_class)
            rows.append(
                {
                    "product": product,
                    "day_class": day_class,
                    "precision": p,
                    "recall": r,
                    "n_pred": n_pred,
                    "n_true": n_true,
                    "tp": tp_p,
                }
            )
    return pd.DataFrame(rows)


def compute_days_metrics(merged: pd.DataFrame) -> dict[str, Any]:
    if merged.empty:
        return {"n": 0, "error": "no merged rows"}

    variants = sorted(merged["variant"].unique()) if "variant" in merged.columns else [3]
    report: dict[str, Any] = {"n": len(merged), "by_variant": {}}

    for variant in variants:
        sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
        vreport: dict[str, Any] = {"n": len(sub), "products": {}}
        day_accs: list[float] = []
        dir_accs: list[float] = []
        per_class_df = _per_day_class_accuracy(sub, variant=int(variant))
        per_dir_df = _per_day_class_direction_accuracy(sub, variant=int(variant))
        per_prec_df = _per_day_class_direction_precision(sub, variant=int(variant))
        per_days_pr_df = _per_days_count_precision_recall(sub, variant=int(variant))

        for product in PRODUCTS:
            true_dir = sub[f"true_dir_{product}"]
            pred_dir = sub[f"pred_dir_{product}"]
            true_days = sub[f"true_days_{product}"].astype(int)
            pred_days = _normalized_pred_days(sub, product)

            dir_acc = float((true_dir == pred_dir).mean())
            days_acc = _days_accuracy(true_days, pred_days)
            dir_accs.append(dir_acc)
            day_accs.append(days_acc)
            vreport["products"][product] = {
                "direction_accuracy": dir_acc,
                "days_class_accuracy": days_acc,
                "true_days_distribution": _class_distribution(true_days),
                "pred_days_distribution": _class_distribution(pred_days),
                "per_day_class_accuracy": {
                    str(row["true_days"]): {
                        "accuracy": row["accuracy"],
                        "n": int(row["n"]),
                    }
                    for _, row in per_class_df[per_class_df["product"] == product].iterrows()
                },
                "per_day_class_direction_accuracy": {
                    str(row["true_days"]): {
                        "direction_accuracy": row["direction_accuracy"],
                        "n": int(row["n"]),
                        "true_up": int(row["true_up"]),
                        "true_flat": int(row["true_flat"]),
                        "true_down": int(row["true_down"]),
                        "pred_up": int(row["pred_up"]),
                        "pred_flat": int(row["pred_flat"]),
                        "pred_down": int(row["pred_down"]),
                    }
                    for _, row in per_dir_df[per_dir_df["product"] == product].iterrows()
                },
                "per_day_class_direction_precision": {
                    str(row["true_days"]): {
                        "precision_flat": row["precision_flat"],
                        "n_pred_flat": int(row["n_pred_flat"]),
                        "precision_up": row["precision_up"],
                        "n_pred_up": int(row["n_pred_up"]),
                        "precision_down": row["precision_down"],
                        "n_pred_down": int(row["n_pred_down"]),
                        "recall_flat": row["recall_flat"],
                        "n_true_flat": int(row["n_true_flat"]),
                        "recall_up": row["recall_up"],
                        "n_true_up": int(row["n_true_up"]),
                        "recall_down": row["recall_down"],
                        "n_true_down": int(row["n_true_down"]),
                    }
                    for _, row in per_prec_df[per_prec_df["product"] == product].iterrows()
                },
                "days_count_precision_recall": {
                    str(int(row["day_class"])): {
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "n_pred": int(row["n_pred"]),
                        "n_true": int(row["n_true"]),
                        "tp": int(row["tp"]),
                    }
                    for _, row in per_days_pr_df[per_days_pr_df["product"] == product].iterrows()
                },
            }

        vreport["direction_accuracy_mean"] = float(np.mean(dir_accs))
        vreport["days_class_accuracy_mean"] = float(np.mean(day_accs))
        report["by_variant"][str(variant)] = vreport

    return report


def plot_days_class_accuracy(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
) -> Path:
    df = _per_day_class_accuracy(merged, variant=variant)
    day_classes = list(range(5))
    x = np.arange(len(day_classes))
    width = 0.25
    colors = {"urea": "#1f77b4", "dap": "#ff7f0e", "mop": "#2ca02c"}
    labels_ru = {"urea": "Urea", "dap": "DAP", "mop": "MOP"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, product in enumerate(PRODUCTS):
        sub = df[df["product"] == product].set_index("true_days").reindex(day_classes)
        accs = sub["accuracy"].tolist()
        ns = sub["n"].tolist()
        bars = ax.bar(
            x + (i - 1) * width,
            [a if a is not None else 0 for a in accs],
            width,
            label=labels_ru[product],
            color=colors[product],
            alpha=0.85,
        )
        for bar, acc, n in zip(bars, accs, ns):
            if acc is None or n == 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    f"n=0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.0%}\nn={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in day_classes])
    ax.set_xlabel("Реальное число дней роста/падения")
    ax.set_ylabel("Accuracy (доля верных предсказаний класса дней)")
    ax.set_title(f"v{variant}: accuracy по классам дней для каждого удобрения")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.2, linewidth=1)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_days_direction_accuracy(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
) -> Path:
    df = _per_day_class_direction_accuracy(merged, variant=variant)
    day_classes = list(range(5))
    x = np.arange(len(day_classes))
    width = 0.25
    colors = {"urea": "#1f77b4", "dap": "#ff7f0e", "mop": "#2ca02c"}
    labels_ru = {"urea": "Urea", "dap": "DAP", "mop": "MOP"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, product in enumerate(PRODUCTS):
        sub = df[df["product"] == product].set_index("true_days").reindex(day_classes)
        accs = sub["direction_accuracy"].tolist()
        ns = sub["n"].tolist()
        bars = ax.bar(
            x + (i - 1) * width,
            [a if a is not None else 0 for a in accs],
            width,
            label=labels_ru[product],
            color=colors[product],
            alpha=0.85,
        )
        for bar, acc, n, day_class in zip(bars, accs, ns, day_classes):
            if acc is None or n == 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "n=0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666",
                )
            else:
                row = sub.loc[day_class]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.0%}\nn={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in day_classes])
    ax.set_xlabel("Реальное число дней роста/падения")
    ax.set_ylabel("Accuracy направления (up / flat / down)")
    ax.set_title(f"v{variant}: accuracy направления по классам дней для каждого удобрения")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.2, linewidth=1)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_days_count_pr(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
    metric: str = "precision",
) -> Path:
    df = _per_days_count_precision_recall(merged, variant=variant)
    day_classes = list(range(5))
    x = np.arange(len(day_classes))
    width = 0.25
    colors = {"urea": "#1f77b4", "dap": "#ff7f0e", "mop": "#2ca02c"}
    labels_ru = {"urea": "Urea", "dap": "DAP", "mop": "MOP"}

    if metric == "recall":
        value_col = "recall"
        count_col = "n_true"
        ylabel = "Recall (полнота)"
        title_metric = "recall — доля угаданных среди реальных"
    else:
        value_col = "precision"
        count_col = "n_pred"
        ylabel = "Precision (точность)"
        title_metric = "precision — доля верных среди предсказаний"

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, product in enumerate(PRODUCTS):
        sub = df[df["product"] == product].set_index("day_class").reindex(day_classes)
        values = sub[value_col].tolist()
        counts = sub[count_col].tolist()
        bars = ax.bar(
            x + (i - 1) * width,
            [v if v is not None else 0 for v in values],
            width,
            label=labels_ru[product],
            color=colors[product],
            alpha=0.85,
        )
        for bar, val, cnt in zip(bars, values, counts):
            if val is None or cnt == 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "n=0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.0%}\nn={cnt}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in day_classes])
    ax.set_xlabel("Класс числа дней (0 = flat / нет серии)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"v{variant}: {title_metric} класса дней для каждого удобрения")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_days_count_precision(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
) -> Path:
    return plot_days_count_pr(merged, output_path, variant=variant, metric="precision")


def plot_days_count_recall(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
) -> Path:
    return plot_days_count_pr(merged, output_path, variant=variant, metric="recall")


def plot_days_direction_pr(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
    flat_threshold: float = 0.01,
    metric: str = "precision",
) -> Path:
    df = _per_day_class_direction_precision(merged, variant=variant)
    day_classes = list(range(5))
    x = np.arange(len(day_classes))
    width = 0.25
    dir_colors = {"flat": "#7f7f7f", "up": "#2ca02c", "down": "#d62728"}
    dir_labels = {"flat": "flat", "up": "up", "down": "down"}
    labels_ru = {"urea": "Urea", "dap": "DAP", "mop": "MOP"}

    if metric == "recall":
        value_cols = {"flat": "recall_flat", "up": "recall_up", "down": "recall_down"}
        count_cols = {"flat": "n_true_flat", "up": "n_true_up", "down": "n_true_down"}
        ylabel = "Recall (полнота)"
        title_metric = "recall (полнота)"
    else:
        value_cols = {"flat": "precision_flat", "up": "precision_up", "down": "precision_down"}
        count_cols = {"flat": "n_pred_flat", "up": "n_pred_up", "down": "n_pred_down"}
        ylabel = "Precision (точность)"
        title_metric = "precision"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, product in zip(axes, PRODUCTS):
        sub = df[df["product"] == product].set_index("true_days").reindex(day_classes)
        for i, direction in enumerate(["flat", "up", "down"]):
            values = sub[value_cols[direction]].tolist()
            counts = sub[count_cols[direction]].tolist()
            bars = ax.bar(
                x + (i - 1) * width,
                [v if v is not None else 0 for v in values],
                width,
                label=dir_labels[direction],
                color=dir_colors[direction],
                alpha=0.85,
            )
            for bar, val, cnt in zip(bars, values, counts):
                if val is None or cnt == 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.02,
                        "n=0",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#666",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.0%}\nn={cnt}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        ax.set_title(labels_ru[product])
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in day_classes])
        ax.set_xlabel("Реальное число дней")
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"v{variant}: {title_metric} направления по классам дней (flat порог {flat_threshold * 100:.0f}%)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_days_direction_precision(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
    flat_threshold: float = 0.01,
) -> Path:
    return plot_days_direction_pr(
        merged, output_path, variant=variant, flat_threshold=flat_threshold, metric="precision"
    )


def plot_days_direction_recall(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
    flat_threshold: float = 0.01,
) -> Path:
    return plot_days_direction_pr(
        merged, output_path, variant=variant, flat_threshold=flat_threshold, metric="recall"
    )


def plot_days_scatter(
    merged: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
) -> Path:
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    norm_pred = {p: _normalized_pred_days(sub, p) for p in PRODUCTS}
    points: list[tuple[int, int, str]] = []
    for product in PRODUCTS:
        for idx, row in sub.iterrows():
            points.append(
                (
                    int(row[f"true_days_{product}"]),
                    int(norm_pred[product].loc[idx]),
                    product,
                )
            )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"urea": "#1f77b4", "dap": "#ff7f0e", "mop": "#2ca02c"}
    for true_d, pred_d, product in points:
        ax.scatter(
            true_d,
            pred_d,
            c=colors[product],
            alpha=0.55,
            s=40,
            edgecolors="white",
            linewidths=0.5,
            label=product if product not in ax.get_legend_handles_labels()[1] else "",
        )

    lim = 4.5
    ax.plot([0, lim], [0, lim], "k--", alpha=0.35, linewidth=1, label="идеал")
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.3, lim)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Реальное число дней роста/падения")
    ax.set_ylabel("Предсказанное число дней")
    ax.set_title(f"v{variant}: реальные vs предсказанные дни (n={len(points)} точек)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_days_comparison(report: dict[str, Any], output_path: Path | str) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, vrep in report.get("by_variant", {}).items():
        rows.append(
            {
                "variant": int(variant),
                "n": vrep["n"],
                "direction_accuracy_mean": vrep["direction_accuracy_mean"],
                "days_class_accuracy_mean": vrep["days_class_accuracy_mean"],
                **{
                    f"{product}_dir_acc": vrep["products"][product]["direction_accuracy"]
                    for product in PRODUCTS
                },
                **{
                    f"{product}_days_acc": vrep["products"][product]["days_class_accuracy"]
                    for product in PRODUCTS
                },
            }
        )
    comparison = pd.DataFrame(rows).sort_values("variant")
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    return comparison


def run_days_analysis(
    merged: pd.DataFrame,
    output_dir: Path | str,
    variant: int = 3,
    flat_threshold: float = 0.01,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    report = compute_days_metrics(merged)
    report["flat_threshold"] = flat_threshold
    report["flat_threshold_pct"] = flat_threshold * 100
    metrics_path = out_dir / "days_metrics.json"
    metrics_path.write_text(
        json.dumps({"test": report, "flat_threshold_pct": flat_threshold * 100}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison_path = out_dir / "days_comparison.csv"
    save_days_comparison(report, comparison_path)

    scatter_path = plot_days_scatter(merged, out_dir / "days_scatter_plot.png", variant=variant)
    class_acc_path = plot_days_class_accuracy(
        merged, out_dir / "days_class_accuracy_plot.png", variant=variant
    )
    per_class_df = _per_day_class_accuracy(merged, variant=variant)
    per_class_csv = out_dir / "days_class_accuracy.csv"
    per_class_df.to_csv(per_class_csv, index=False, encoding="utf-8-sig")

    per_dir_df = _per_day_class_direction_accuracy(merged, variant=variant)
    per_dir_csv = out_dir / "days_direction_accuracy.csv"
    per_dir_df.to_csv(per_dir_csv, index=False, encoding="utf-8-sig")
    dir_acc_path = plot_days_direction_accuracy(
        merged, out_dir / "days_direction_accuracy_plot.png", variant=variant
    )

    per_prec_df = _per_day_class_direction_precision(merged, variant=variant)
    per_prec_csv = out_dir / "days_direction_precision.csv"
    per_prec_df.to_csv(per_prec_csv, index=False, encoding="utf-8-sig")
    dir_prec_path = plot_days_direction_precision(
        merged,
        out_dir / "days_direction_precision_plot.png",
        variant=variant,
        flat_threshold=flat_threshold,
    )
    dir_rec_path = plot_days_direction_recall(
        merged,
        out_dir / "days_direction_recall_plot.png",
        variant=variant,
        flat_threshold=flat_threshold,
    )

    per_days_pr_df = _per_days_count_precision_recall(merged, variant=variant)
    per_days_pr_csv = out_dir / "days_count_precision_recall.csv"
    per_days_pr_df.to_csv(per_days_pr_csv, index=False, encoding="utf-8-sig")
    days_prec_path = plot_days_count_precision(
        merged, out_dir / "days_count_precision_plot.png", variant=variant
    )
    days_rec_path = plot_days_count_recall(
        merged, out_dir / "days_count_recall_plot.png", variant=variant
    )

    scatter_rows: list[dict] = []
    sub = merged[merged["variant"] == variant] if "variant" in merged.columns else merged
    norm_pred = {p: _normalized_pred_days(sub, p) for p in PRODUCTS}
    for idx, row in sub.iterrows():
        for product in PRODUCTS:
            scatter_rows.append(
                {
                    "date": row["date"],
                    "source": row["source"],
                    "headline": row["headline"],
                    "product": product,
                    "true_dir": row[f"true_dir_{product}"],
                    "pred_dir": row[f"pred_dir_{product}"],
                    "true_days": int(row[f"true_days_{product}"]),
                    "pred_days": int(norm_pred[product].loc[idx]),
                }
            )
    scatter_csv = out_dir / "days_scatter_data.csv"
    pd.DataFrame(scatter_rows).to_csv(scatter_csv, index=False, encoding="utf-8-sig")

    return {
        "metrics": metrics_path,
        "comparison": comparison_path,
        "plot": scatter_path,
        "class_accuracy_plot": class_acc_path,
        "class_accuracy_csv": per_class_csv,
        "direction_accuracy_plot": dir_acc_path,
        "direction_accuracy_csv": per_dir_csv,
        "direction_precision_plot": dir_prec_path,
        "direction_precision_csv": per_prec_csv,
        "direction_recall_plot": dir_rec_path,
        "days_count_precision_plot": days_prec_path,
        "days_count_recall_plot": days_rec_path,
        "days_count_precision_recall_csv": per_days_pr_csv,
        "scatter_csv": scatter_csv,
        "report": report,
    }


def _merge_predictions_with_days_gt(
    predictions: pd.DataFrame,
    scope: pd.DataFrame,
    prices_path: str,
    flat_threshold: float,
    max_days: int = 4,
) -> pd.DataFrame:
    computed = compute_day_streaks_for_news_df(
        scope,
        prices_path,
        max_days=max_days,
        flat_threshold=flat_threshold,
    )
    rows: list[dict] = []
    for _, row in computed.iterrows():
        item = {
            "date": pd.Timestamp(row["date"]),
            "source": row["source"],
            "headline": row["headline"],
        }
        for product in PRODUCTS:
            s = row["streaks"][product]
            item[f"true_dir_{product}"] = s["direction"]
            item[f"true_days_{product}"] = s["days"]
        rows.append(item)
    ground_truth = pd.DataFrame(rows)
    pred_cols = [c for c in predictions.columns if not c.startswith("true_")]
    return predictions[pred_cols].merge(
        ground_truth,
        on=["date", "source", "headline"],
        how="inner",
    )


def compare_flat_thresholds(
    predictions: pd.DataFrame,
    scope: pd.DataFrame,
    prices_path: str,
    thresholds: list[float],
    variant: int = 3,
    max_days: int = 4,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for th in thresholds:
        merged = _merge_predictions_with_days_gt(
            predictions, scope, prices_path, flat_threshold=th, max_days=max_days
        )
        prec = _per_day_class_direction_precision(merged, variant=variant)
        acc = _per_day_class_direction_accuracy(merged, variant=variant)
        days_acc = _per_day_class_accuracy(merged, variant=variant)
        prec = prec.merge(
            acc[["product", "true_days", "direction_accuracy"]],
            on=["product", "true_days"],
            how="left",
        )
        prec = prec.merge(
            days_acc[["product", "true_days", "accuracy"]].rename(
                columns={"accuracy": "days_class_accuracy"}
            ),
            on=["product", "true_days"],
            how="left",
        )
        prec["flat_threshold"] = th
        prec["flat_threshold_pct"] = th * 100
        frames.append(prec)
    return pd.concat(frames, ignore_index=True)


def plot_flat_threshold_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path | str,
    variant: int = 3,
    metric: str = "precision",
) -> Path:
    thresholds = sorted(comparison_df["flat_threshold_pct"].unique())
    day_classes = list(range(5))
    x = np.arange(len(day_classes))
    width = 0.12
    dir_colors = {"flat": "#7f7f7f", "up": "#2ca02c", "down": "#d62728"}
    dir_labels = {"flat": "flat", "up": "up", "down": "down"}
    labels_ru = {"urea": "Urea", "dap": "DAP", "mop": "MOP"}

    if metric == "recall":
        value_cols = {"flat": "recall_flat", "up": "recall_up", "down": "recall_down"}
        count_cols = {"flat": "n_true_flat", "up": "n_true_up", "down": "n_true_down"}
        ylabel = "Recall (полнота)"
        title_metric = "recall (полнота)"
    else:
        value_cols = {"flat": "precision_flat", "up": "precision_up", "down": "precision_down"}
        count_cols = {"flat": "n_pred_flat", "up": "n_pred_up", "down": "n_pred_down"}
        ylabel = "Precision"
        title_metric = "precision"

    n_th = len(thresholds)
    fig, axes = plt.subplots(n_th, 3, figsize=(15, 4.5 * n_th), sharey=True)
    if n_th == 1:
        axes = np.array([axes])

    for row, th_pct in enumerate(thresholds):
        th_df = comparison_df[np.isclose(comparison_df["flat_threshold_pct"], th_pct)]
        for col, product in enumerate(PRODUCTS):
            ax = axes[row, col]
            sub = th_df[th_df["product"] == product].set_index("true_days").reindex(day_classes)
            for dir_i, direction in enumerate(["flat", "up", "down"]):
                values = sub[value_cols[direction]].tolist()
                counts = sub[count_cols[direction]].tolist()
                offset = (dir_i - 1) * width
                bars = ax.bar(
                    x + offset,
                    [v if v is not None else 0 for v in values],
                    width,
                    label=dir_labels[direction],
                    color=dir_colors[direction],
                    alpha=0.85,
                )
                for bar, val, cnt in zip(bars, values, counts):
                    if val is None or cnt == 0:
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
            ax.set_title(f"{labels_ru[product]} — flat {th_pct:.0f}%")
            ax.set_xticks(x)
            ax.set_xticklabels([str(d) for d in day_classes])
            ax.set_xlabel("Реальное число дней")
            ax.set_ylim(0, 1.12)
            ax.grid(True, axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"v{variant}: {title_metric} up/flat/down — сравнение порогов flat",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_flat_threshold_comparison(
    predictions: pd.DataFrame,
    scope: pd.DataFrame,
    prices_path: str,
    output_dir: Path | str,
    thresholds: list[float] | None = None,
    variant: int = 3,
    max_days: int = 4,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [0.01, 0.02]
    out_dir = Path(output_dir)
    comparison_df = compare_flat_thresholds(
        predictions, scope, prices_path, thresholds, variant=variant, max_days=max_days
    )
    csv_path = out_dir / "days_flat_threshold_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = plot_flat_threshold_comparison(
        comparison_df,
        out_dir / "days_flat_threshold_comparison_plot.png",
        variant=variant,
        metric="precision",
    )
    recall_plot_path = plot_flat_threshold_comparison(
        comparison_df,
        out_dir / "days_flat_threshold_recall_plot.png",
        variant=variant,
        metric="recall",
    )

    summary_rows: list[dict] = []
    for th in thresholds:
        merged = _merge_predictions_with_days_gt(
            predictions, scope, prices_path, flat_threshold=th, max_days=max_days
        )
        report = compute_days_metrics(merged)
        vrep = report.get("by_variant", {}).get(str(variant), {})
        summary_rows.append(
            {
                "flat_threshold_pct": th * 100,
                "direction_accuracy_mean": vrep.get("direction_accuracy_mean"),
                "days_class_accuracy_mean": vrep.get("days_class_accuracy_mean"),
                **{
                    f"{p}_dir_acc": vrep["products"][p]["direction_accuracy"]
                    for p in PRODUCTS
                    if vrep.get("products")
                },
                **{
                    f"{p}_days_acc": vrep["products"][p]["days_class_accuracy"]
                    for p in PRODUCTS
                    if vrep.get("products")
                },
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "days_flat_threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "comparison_csv": csv_path,
        "comparison_plot": plot_path,
        "comparison_recall_plot": recall_plot_path,
        "summary_csv": summary_path,
        "comparison_df": comparison_df,
        "summary_df": summary_df,
    }
