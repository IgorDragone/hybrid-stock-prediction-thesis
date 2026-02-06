from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Purged walk-forward cross-validation configuration.

    Notes
    -----
    - embargo_days is in *calendar days* (simple + robust). For trading-day embargo,
      you can switch to an index-based approach later if needed.
    """
    date_col: str = "date"
    train_years: int = 3
    test_months: int = 6
    embargo_days: int = 126
    min_train_days: int = 252


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return dt + pd.DateOffset(months=months)


def generate_purged_walk_forward_splits(
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
    start_test_date: Optional[pd.Timestamp] = None,
    end_test_date: Optional[pd.Timestamp] = None,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Yield (train_df, test_df, info) for purged walk-forward CV with embargo.

    Train window: [train_start, train_end]
    Embargo:      (train_end, test_start]
    Test window:  (test_start, test_end]

    This prevents label overlap leakage when targets use forward returns.
    """
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column: {cfg.date_col}")

    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d = d.sort_values(cfg.date_col)

    dates = d[cfg.date_col].drop_duplicates().sort_values()
    if len(dates) < cfg.min_train_days:
        raise ValueError("Not enough distinct dates to create splits.")

    min_date = dates.iloc[0]
    max_date = dates.iloc[-1]

    if start_test_date is None:
        start_test_date = min_date + pd.DateOffset(years=cfg.train_years) + pd.Timedelta(days=cfg.embargo_days)
    if end_test_date is None:
        end_test_date = max_date

    test_start = pd.Timestamp(start_test_date)
    while test_start < end_test_date:
        train_end = test_start - pd.Timedelta(days=cfg.embargo_days)
        train_start = train_end - pd.DateOffset(years=cfg.train_years)
        test_end = _add_months(test_start, cfg.test_months)

        if train_start < min_date:
            train_start = min_date
        if test_end > max_date:
            test_end = max_date

        train_mask = (d[cfg.date_col] >= train_start) & (d[cfg.date_col] <= train_end)
        test_mask = (d[cfg.date_col] > test_start) & (d[cfg.date_col] <= test_end)

        train_df = d.loc[train_mask].copy()
        test_df = d.loc[test_mask].copy()

        if len(test_df) == 0:
            break
        if len(train_df[cfg.date_col].drop_duplicates()) < cfg.min_train_days:
            test_start = _add_months(test_start, cfg.test_months)
            continue

        info = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "embargo_days": cfg.embargo_days,
        }

        if train_df[cfg.date_col].max() + pd.Timedelta(days=cfg.embargo_days) > test_df[cfg.date_col].min():
            raise AssertionError("Embargo violation: train and test overlap after embargo.")

        yield train_df, test_df, info

        test_start = _add_months(test_start, cfg.test_months)
