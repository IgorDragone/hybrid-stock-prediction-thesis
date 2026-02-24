from __future__ import annotations

import numpy as np
import pandas as pd


def mean_monthly_spearman_ic(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    ret_col: str,
) -> float:
    """Mean monthly cross-sectional Spearman correlation between score and forward returns."""
    ic = (
        df.groupby(date_col)
        .apply(lambda x: x[score_col].corr(x[ret_col], method="spearman"))
    )
    return float(ic.mean())


def top_k_hit_rate(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    ret_col: str,
    k: int,
    ticker_col: str = "ticker",
) -> float:
    """Fraction of overlap between predicted top-K and realized top-K each month."""
    def _hit(x: pd.DataFrame) -> float:
        pred = x.nlargest(k, score_col)[ticker_col].tolist()
        actual = x.nlargest(k, ret_col)[ticker_col].tolist()
        if not pred:
            return np.nan
        return len(set(pred) & set(actual)) / k

    hit = df.groupby(date_col).apply(_hit)
    return float(hit.mean())
