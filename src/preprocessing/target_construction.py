#src/preprocessing/target_construction.py
"""
Target construction for stock prediction.

This module provides functionality to create leakage-free targets for
supervised learning based on forward returns and cross-sectional ranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
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
    Configuration for target construction.

    Attributes:
        date_col: Name of the date column in the panel DataFrame.
        ticker_col: Name of the ticker/asset identifier column.
        price_col: Name of the price column used to compute forward returns.
        horizon_3m: Horizon in trading days for 3-month forward return.
        horizon_6m: Horizon in trading days for 6-month forward return.
        bot_q: Lower quantile threshold for sell signal.
        top_q: Upper quantile threshold for buy signal.
        min_points_cs: Minimum non-NaN points in cross-section to compute quantiles.
        fwd_ret_3m_col: Name of the output column for 3-month forward return.
        fwd_ret_6m_col: Name of the output column for 6-month forward return.
        fwd_score_col: Name of the output column for combined forward return score.
        target_col: Name of the output column for the final target labels.
    """
    # Panel keys
    date_col: str = "date"
    ticker_col: str = "ticker"

    # Price column used to compute forward returns
    price_col: str = "adj_close"

    # Horizons (trading days)
    horizon_3m: int = 63
    horizon_6m: int = 126

    # Quantile thresholds for cross-sectional labels
    # bottom <= bot_q  -> sell (-1)
    # top    >= top_q  -> buy  (+1)
    bot_q: float = 0.30
    top_q: float = 0.70

    # Minimum non-NaN points in cross-section to compute quantiles
    min_points_cs: int = 5

    # Output column names
    fwd_ret_3m_col: str = "fwd_ret_3m"
    fwd_ret_6m_col: str = "fwd_ret_6m"
    fwd_score_col: str = "fwd_ret_3_6m"
    target_col: str = "target"


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    # Ensure that required columns are present in the DataFrame
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def construct_target(df: pd.DataFrame, config: Optional[TargetConfig] = None) -> pd.DataFrame:
    """
    Construct leakage-free targets using forward returns and cross-sectional ranking.

    Steps:
    1) compute forward returns at 3m and 6m horizons per ticker
    2) combine them into a single continuous score (mean)
    3) per date, compute cross-sectional quantile thresholds
    4) assign labels: sell=-1, hold=0, buy=+1
       rows without forward score remain NaN (cannot be used for training)

    Args:
        df (pd.DataFrame): Input panel DataFrame with price data.
        config (TargetConfig | None): Optional TargetConfig instance with parameters.

    Returns:
        pd.DataFrame: DataFrame with additional columns for forward returns, score, and target labels.
    """
    cfg = config or TargetConfig()
    logger.info("Construct target: %d rows, %d columns", df.shape[0], df.shape[1])
    out = df.copy()

    _require_cols(out, (cfg.date_col, cfg.ticker_col, cfg.price_col))
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])

    # Sort for groupby-shift correctness
    out = out.sort_values([cfg.ticker_col, cfg.date_col]).reset_index(drop=True)

    # Forward prices
    fwd_p3 = out.groupby(cfg.ticker_col)[cfg.price_col].shift(-cfg.horizon_3m)
    fwd_p6 = out.groupby(cfg.ticker_col)[cfg.price_col].shift(-cfg.horizon_6m)

    # Forward returns
    out[cfg.fwd_ret_3m_col] = fwd_p3 / out[cfg.price_col] - 1.0
    out[cfg.fwd_ret_6m_col] = fwd_p6 / out[cfg.price_col] - 1.0

    # Combined score (3–6m)
    out[cfg.fwd_score_col] = out[[cfg.fwd_ret_3m_col, cfg.fwd_ret_6m_col]].mean(axis=1)

    # Cross-sectional thresholds per date
    def q_if_enough(s: pd.Series, q: float) -> float:
        if s.notna().sum() < cfg.min_points_cs:
            return np.nan
        return float(s.quantile(q))

    q_hi = out.groupby(cfg.date_col)[cfg.fwd_score_col].transform(lambda s: q_if_enough(s, cfg.top_q))
    q_lo = out.groupby(cfg.date_col)[cfg.fwd_score_col].transform(lambda s: q_if_enough(s, cfg.bot_q))

    # Labels
    out[cfg.target_col] = 0.0  # float temporarily to allow NaN
    out.loc[out[cfg.fwd_score_col] >= q_hi, cfg.target_col] = 1.0
    out.loc[out[cfg.fwd_score_col] <= q_lo, cfg.target_col] = -1.0

    # If score missing -> target missing (cannot train)
    out.loc[out[cfg.fwd_score_col].isna(), cfg.target_col] = np.nan

    logger.info("Target construction complete: %d rows, %d columns", out.shape[0], out.shape[1])
    return out


def make_model_ready(df_with_target: pd.DataFrame, config: Optional[TargetConfig] = None) -> pd.DataFrame:
    """
    Return a model-ready panel: keep only rows where target is available and cast target to int8.

    Args:
        df_with_target (pd.DataFrame): DataFrame with constructed target column.
        config (TargetConfig | None): Optional TargetConfig instance.
    Returns:
        pd.DataFrame: DataFrame filtered to rows with non-NaN target and target cast to int8.
    """
    cfg = config or TargetConfig()
    _require_cols(df_with_target, (cfg.target_col,))
    out = df_with_target.dropna(subset=[cfg.target_col]).copy()
    out[cfg.target_col] = out[cfg.target_col].astype("int8")
    return out
