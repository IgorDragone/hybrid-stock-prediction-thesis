from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Expanding walk-forward CV for monthly EOM panels with embargo."""
    date_col: str = "date"
    train_years: int = 10
    test_months: int = 12
    embargo_months: int = 3
    min_train_months: int = 24


def _add_months(period: pd.Period, months: int) -> pd.Period:
    return period + months


def generate_expanding_walk_forward_splits(
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
    start_test_month: Optional[pd.Period] = None,
    end_test_month: Optional[pd.Period] = None,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Yield (train_df, test_df, info) for expanding walk-forward with embargo.

    Train window: [start, train_end]
    Embargo:      (train_end, test_start)
    Test window:  [test_start, test_end]
    """
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column: {cfg.date_col}")

    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d["_month"] = d[cfg.date_col].dt.to_period("M")
    d = d.sort_values(cfg.date_col)

    months = d["_month"].drop_duplicates().sort_values()
    if len(months) < cfg.min_train_months:
        raise ValueError("Not enough distinct months to create splits.")

    min_month = months.iloc[0]
    max_month = months.iloc[-1]

    if start_test_month is None:
        start_test_month = _add_months(min_month, cfg.train_years * 12 + cfg.embargo_months)
    if end_test_month is None:
        end_test_month = max_month

    test_start = pd.Period(start_test_month, freq="M")
    while test_start <= end_test_month:
        train_end = _add_months(test_start, -cfg.embargo_months)
        test_end = _add_months(test_start, cfg.test_months - 1)
        if test_end > max_month:
            test_end = max_month

        train_mask = d["_month"] <= train_end
        test_mask = (d["_month"] >= test_start) & (d["_month"] <= test_end)

        train_df = d.loc[train_mask].copy()
        test_df = d.loc[test_mask].copy()
        if len(test_df) == 0:
            break
        if train_df["_month"].nunique() < cfg.min_train_months:
            test_start = _add_months(test_start, cfg.test_months)
            continue

        info = {
            "train_start": train_df[cfg.date_col].min(),
            "train_end": train_df[cfg.date_col].max(),
            "test_start": test_df[cfg.date_col].min(),
            "test_end": test_df[cfg.date_col].max(),
            "embargo_months": cfg.embargo_months,
        }

        yield train_df.drop(columns=["_month"]), test_df.drop(columns=["_month"]), info

        test_start = _add_months(test_start, cfg.test_months)
