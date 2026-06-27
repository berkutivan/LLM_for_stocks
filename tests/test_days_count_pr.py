import pandas as pd

from src.eval.days_metrics import (
    _per_days_count_precision_recall,
    normalize_pred_days_value,
)


def test_normalize_pred_days_legacy_weeks_times_seven():
    assert normalize_pred_days_value(7, weeks=1, direction="up") == 1
    assert normalize_pred_days_value(14, weeks=2, direction="down") == 2
    assert normalize_pred_days_value(0, weeks=0, direction="flat") == 0


def test_days_count_precision_recall_per_class():
    merged = pd.DataFrame(
        {
            "variant": [3, 3, 3],
            "true_days_urea": [1, 1, 2],
            "pred_days_urea": [1, 2, 2],
            "pred_weeks_urea": [0, 0, 0],
            "pred_dir_urea": ["up", "up", "down"],
            "true_days_dap": [1, 1, 1],
            "pred_days_dap": [1, 1, 1],
            "pred_weeks_dap": [0, 0, 0],
            "pred_dir_dap": ["up", "up", "up"],
            "true_days_mop": [0, 0, 0],
            "pred_days_mop": [0, 0, 0],
            "pred_weeks_mop": [0, 0, 0],
            "pred_dir_mop": ["flat", "flat", "flat"],
        }
    )
    df = _per_days_count_precision_recall(merged, variant=3)
    urea_1 = df[(df["product"] == "urea") & (df["day_class"] == 1)].iloc[0]
    assert urea_1["precision"] == 1.0
    assert urea_1["recall"] == 0.5
    assert urea_1["n_pred"] == 1
    assert urea_1["n_true"] == 2
    urea_2 = df[(df["product"] == "urea") & (df["day_class"] == 2)].iloc[0]
    assert urea_2["precision"] == 0.5
    assert urea_2["recall"] == 1.0
    assert urea_2["n_pred"] == 2
    assert urea_2["n_true"] == 1
