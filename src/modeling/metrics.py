from __future__ import annotations

import numpy as np
import pandas as pd


def mean_daily_spearman_ic(df: pd.DataFrame, date_col: str, score_col: str, ret_col: str) -> float:
    """Mean daily cross-sectional Spearman correlation between score and forward returns."""
    ic = df.groupby(date_col).apply(lambda x: x[score_col].corr(x[ret_col], method="spearman"))
    return float(ic.mean())


def top_bottom_spread(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    ret_col: str,
    q: float = 0.3,
) -> float:
    """Average (Top q mean - Bottom q mean) return using score-based ranking per date."""
    q_low = df.groupby(date_col)[score_col].transform(lambda x: x.quantile(q))
    q_high = df.groupby(date_col)[score_col].transform(lambda x: x.quantile(1 - q))

    sig = np.where(df[score_col] >= q_high, 1, np.where(df[score_col] <= q_low, -1, 0))
    top = df.loc[sig == 1, ret_col].mean()
    bot = df.loc[sig == -1, ret_col].mean()
    return float(top - bot)
