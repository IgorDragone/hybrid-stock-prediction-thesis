#src/preprocessing/target_construction.py
"""
Target construction for stock prediction.

This module provides functionality to create leakage-free targets for
supervised learning based on forward returns and cross-sectional demeaning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class TargetConfig:
    """
    Configuration for target construction (EOM targets).

    Attributes:
        date_col: Name of the date column in the panel DataFrame.
        ticker_col: Name of the ticker/asset identifier column.
        price_col: Name of the price column used to compute forward returns.
        horizon_1m: Horizon in periods for 1-month forward return.
        horizon_3m: Horizon in periods for 3-month forward return.
        horizon_6m: Horizon in periods for 6-month forward return (analysis only).
        ret_1m_col: Name of the output column for the 1-month raw forward return.
        ret_3m_col: Name of the output column for the 3-month raw forward return.
        ret_6m_col: Name of the output column for the 6-month raw forward return.
        target_1m_col: Name of the output column for the 1-month demeaned target.
        target_3m_col: Name of the output column for the 3-month demeaned target.
        target_col: Name of the primary target column (default: 3-month).
    """
    # Panel keys
    date_col: str = "date"
    ticker_col: str = "ticker"

    # Price column used to compute forward returns
    price_col: str = "adj_close"

    # Horizons (periods). With EOM data, 1 step = 1 month.
    horizon_1m: int = 1
    horizon_3m: int = 3
    horizon_6m: int = 6

    # Output column names
    ret_1m_col: str = "fwd_ret_1m"
    ret_3m_col: str = "fwd_ret_3m"
    ret_6m_col: str = "fwd_ret_6m"
    target_1m_col: str = "target_1m"
    target_3m_col: str = "target_3m"
    target_col: str = "target_3m"


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    # Ensure that required columns are present in the DataFrame
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def construct_target(df: pd.DataFrame, config: Optional[TargetConfig] = None) -> pd.DataFrame:
    """
    Construct leakage-free targets using forward returns and cross-sectional demeaning.

    Steps:
    1) compute forward returns at 1m and 3m horizons per ticker
    2) demean the returns cross-sectionally per date
    3) targets at the tail remain NaN (insufficient forward data)

    Args:
        df (pd.DataFrame): Input panel DataFrame with price data.
        config (TargetConfig | None): Optional TargetConfig instance with parameters.

    Returns:
        pd.DataFrame: DataFrame with additional columns for forward returns and targets.
    """
    cfg = config or TargetConfig()
    logger.info("Construct target: %d rows, %d columns", df.shape[0], df.shape[1])
    out = df.copy()

    _require_cols(out, (cfg.date_col, cfg.ticker_col, cfg.price_col))
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])

    # Sort for groupby-shift correctness
    out = out.sort_values([cfg.ticker_col, cfg.date_col]).reset_index(drop=True)

    # Forward prices (period shift; with EOM data, each row is a month)
    fwd_p1 = out.groupby(cfg.ticker_col)[cfg.price_col].shift(-cfg.horizon_1m)
    fwd_p3 = out.groupby(cfg.ticker_col)[cfg.price_col].shift(-cfg.horizon_3m)
    fwd_p6 = out.groupby(cfg.ticker_col)[cfg.price_col].shift(-cfg.horizon_6m)

    # Raw forward returns
    out[cfg.ret_1m_col] = fwd_p1 / out[cfg.price_col] - 1.0
    out[cfg.ret_3m_col] = fwd_p3 / out[cfg.price_col] - 1.0
    out[cfg.ret_6m_col] = fwd_p6 / out[cfg.price_col] - 1.0

    # Cross-sectional demeaning
    out[cfg.target_1m_col] = out[cfg.ret_1m_col] - out.groupby(cfg.date_col)[cfg.ret_1m_col].transform("mean")
    out[cfg.target_3m_col] = out[cfg.ret_3m_col] - out.groupby(cfg.date_col)[cfg.ret_3m_col].transform("mean")

    logger.info("Target construction complete: %d rows, %d columns", out.shape[0], out.shape[1])
    return out


def make_model_ready(df_with_target: pd.DataFrame, config: Optional[TargetConfig] = None) -> pd.DataFrame:
    """
    Return a model-ready panel: keep only rows where target is available.

    Args:
        df_with_target (pd.DataFrame): DataFrame with constructed target column.
        config (TargetConfig | None): Optional TargetConfig instance.
    Returns:
        pd.DataFrame: DataFrame filtered to rows with non-NaN target.
    """
    cfg = config or TargetConfig()
    _require_cols(df_with_target, (cfg.target_col,))
    out = df_with_target.dropna(subset=[cfg.target_col]).copy()
    return out
