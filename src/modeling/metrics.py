from __future__ import annotations

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
