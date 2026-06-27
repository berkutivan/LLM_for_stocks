from __future__ import annotations

import pandas as pd


def temporal_split(
    news_df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = news_df.sort_values("date").reset_index(drop=True)
    cutoff = int(len(df) * train_ratio)
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()
    return train, test
