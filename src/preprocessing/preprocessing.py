# src/preprocessing/preprocessing.py
"""
Preprocessing utilities for a daily long-format panel dataset.

The module applies domain-aware stabilization (clipping),
macro forward-fill, and feature transformations while preserving
panel integrity (sorted, unique (date, ticker)).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# -----------------------------------------------
# Config dataclasses
# -----------------------------------------------
@dataclass(frozen=True)
class DomainClipConfig:
    """
    Domain-aware clipping rules for economically bounded ratios.

    The bounds reduce the influence of extreme outliers caused by reporting noise,
    denominator effects (e.g., small equity), or data vendor artifacts.

    Attributes:
        growth_bounds (Tuple[float, float]): min/max bounds for growth rates
        margin_bounds (Tuple[float, float]): min/max bounds for margins
        roe_bounds (Tuple[float, float]): min/max bounds for return on equity
        roa_bounds (Tuple[float, float]): min/max bounds for return on assets
        debt_to_equity_bounds (Optional[Tuple[float, float]]): min/max bounds for debt to equity ratio
        interest_coverage_bounds (Optional[Tuple[float, float]]): min/max bounds for interest coverage
        asset_turnover_bounds (Optional[Tuple[float, float]]): min/max bounds for asset turnover
        fcf_assets_bounds (Optional[Tuple[float, float]]): min/max bounds for fcf/assets
    """
    # Growth rates: asymmetrically bounded (base effects)
    growth_bounds: Tuple[float, float] = (-0.5, 2.0)  # [-50%, +200%]

    # Margins: economically bounded-ish. Allow negatives but clip extreme tails.
    margin_bounds: Tuple[float, float] = (-1.0, 1.0)  # [-100%, +100%]

    # ROE/ROA can spike when equity/assets are small/negative. Clip to reasonable ranges.
    roe_bounds: Tuple[float, float] = (-2.0, 2.0)
    roa_bounds: Tuple[float, float] = (-1.0, 1.0)
    delta_roe_bounds: Tuple[float, float] = (-1.0, 1.0)

    # Optional: leverage/ratios can be huge when denominator tiny; clip if desired.
    # Set to None to disable.
    debt_to_equity_bounds: Optional[Tuple[float, float]] = (0.0, 10.0)
    interest_coverage_bounds: Optional[Tuple[float, float]] = (0.0, 20.0)
    asset_turnover_bounds: Optional[Tuple[float, float]] = (0.0, 5.0)
    fcf_assets_bounds: Optional[Tuple[float, float]] = (-0.5, 0.5)


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for panel preprocessing.

    Attributes:
        macro_ffill (bool): Whether to forward-fill macro variables.
        domain_clip (bool): Enable domain-aware clipping of ratios.
        min_non_na_ratio (float): Minimum non-missing ratio to keep a column.
    """
    date_col: str = "date"
    ticker_col: str = "ticker"

    # Column groups
    macro_cols: Sequence[str]
    fundamental_cols: Sequence[str]
    growth_cols: Sequence[str]
    margin_cols: Sequence[str]

    # Steps toggles
    replace_infs_with_nan: bool = True
    macro_ffill: bool = True
    fundamentals_ffill: bool = True

    # Domain clipping
    domain_clip: bool = True
    domain_clip_config: DomainClipConfig = DomainClipConfig()

    # Column-level NaN pruning
    prune_sparse_columns: bool = False
    min_non_na_ratio: float = 0.90

    # Cross-sectional winsorization (applied post-EOM in the pipeline)
    winsorize_fundamentals_cs: bool = True
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    growth_winsor_lower_q: float = 0.02
    growth_winsor_upper_q: float = 0.98


# -----------------------------------------------
# Helper functions
# -----------------------------------------------
def _check_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    # Check that required columns are present in df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    # Ensure date_col is datetime, raise if unparsable
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad = out[out[date_col].isna()]
        raise ValueError(
            f"`{date_col}` contains non-parsable dates (n={len(bad)}). "
            "Fix upstream or coerce before preprocessing."
        )
    return out


def _assert_unique_panel(df: pd.DataFrame, date_col: str, ticker_col: str) -> None:
    # Assert that (date_col, ticker_col) is unique in df
    dup = int(df.duplicated([date_col, ticker_col]).sum())
    if dup:
        raise ValueError(
            f"Found {dup} duplicated rows on ({date_col}, {ticker_col}). "
            "DB build should guarantee uniqueness."
        )


def _replace_infs(df: pd.DataFrame) -> pd.DataFrame:
    # Replace +inf/-inf with NaN in numeric columns
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return out
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _ffill_cols_within_ticker(
    df: pd.DataFrame, ticker_col: str, cols: Sequence[str]
) -> pd.DataFrame:
    """
    Forward-fill selected columns within each ticker over time.

    This is used to propagate the most recent available value (e.g., macro
    features) to subsequent days without leaking information across tickers.
    """
    if not cols:
        return df
    
    out = df.copy()
    out[list(cols)] = out.groupby(ticker_col, group_keys=False)[list(cols)].ffill()

    return out


def _clip_cols(df: pd.DataFrame, cols: Sequence[str], lo: float, hi: float) -> pd.DataFrame:
    # Clip a set of columns to fixed bounds, ignoring missing columns.
    if not cols:
        return df
    out = df.copy()
    existing = [c for c in cols if c in out.columns]
    if existing:
        out[existing] = out[existing].clip(lower=lo, upper=hi)
    return out


def _prune_sparse_columns(df: pd.DataFrame, min_non_na_ratio: float) -> pd.DataFrame:
    # Drop columns with excessive missing values.
    if not (0.0 < min_non_na_ratio <= 1.0):
        raise ValueError("min_non_na_ratio must be in (0, 1].")
    keep = df.columns[df.notna().mean() >= min_non_na_ratio].tolist()
    return df[keep].copy()


def _winsorize_cross_sectional(
    df: pd.DataFrame,
    date_col: str,
    cols: Sequence[str],
    lower_q: float,
    upper_q: float,
    min_points: int = 5,
) -> pd.DataFrame:
    """
    Winsorize selected columns cross-sectionally, date by date.

    For each date, values in each column are clipped to the [lower_q, upper_q]
    quantiles computed across tickers. This stabilizes noisy ratios while
    preserving time-series structure.
    """
    if not cols:
        return df
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1.")
    out = df.copy()

    def _clip_group(group_df: pd.DataFrame) -> pd.DataFrame:
        clipped = group_df.copy()
        for col in cols:
            series = clipped[col]
            # Avoid unstable quantiles when too few observations exist for that date
            if series.notna().sum() < min_points:
                continue
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            clipped[col] = series.clip(lower, upper)
        return clipped

    return out.groupby(date_col, group_keys=False).apply(_clip_group)


def winsorize_fundamentals_cs(
    df: pd.DataFrame,
    *,
    date_col: str,
    fundamental_cols: Sequence[str],
    growth_cols: Sequence[str],
    lower_q: float,
    upper_q: float,
    growth_lower_q: float,
    growth_upper_q: float,
) -> pd.DataFrame:
    """
    Cross-sectional winsorization for fundamentals, intended to run on EOM data.
    """
    cols_default = [c for c in fundamental_cols if c not in growth_cols]
    out = df.copy()
    if cols_default:
        out = _winsorize_cross_sectional(
            out,
            date_col=date_col,
            cols=cols_default,
            lower_q=lower_q,
            upper_q=upper_q,
        )
    if growth_cols:
        out = _winsorize_cross_sectional(
            out,
            date_col=date_col,
            cols=list(growth_cols),
            lower_q=growth_lower_q,
            upper_q=growth_upper_q,
        )
    return out


#-----------------------------------------------
# Main preprocessing function
#-----------------------------------------------
def preprocess_daily_panel(df: pd.DataFrame, config: Optional[PreprocessConfig] = None) -> pd.DataFrame:
    """
    Preprocess a daily long-format panel DataFrame.

    The function applies a sequence of transformations intended to stabilize
    fundamentals and macro features while preserving panel consistency.

    Steps:
        1) Validate required columns and parse dates.
        2) Enforce sorting and uniqueness of (date, ticker).
        3) Replace +/-inf with NaN (numeric columns).
        4) Resolve column groups (explicit lists).
        5) Forward-fill macro columns within each ticker.
        6) Domain-aware clipping of selected ratios (optional).
        7) Prune columns with excessive missing values.

    Args:
        df (pd.DataFrame): Long-format panel with at least (date, ticker).
        config (PreprocessConfig): Preprocessing configuration.

    Returns:
        pd.DataFrame: Processed panel sorted by (date, ticker) with a fresh integer index.
    """
    logger.info("Preprocess panel: %d rows, %d columns", df.shape[0], df.shape[1])
    cfg = config or PreprocessConfig()
    _check_required_columns(df, [cfg.date_col, cfg.ticker_col])

    out = df.copy()
    out = _ensure_datetime(out, cfg.date_col)

    # As we need to forward-fill and do cross-sectional ops, ensure sortedness
    out = out.sort_values([cfg.date_col, cfg.ticker_col])

    # Very quick uniqueness check, as DB build should guarantee this
    _assert_unique_panel(out, cfg.date_col, cfg.ticker_col)

    # Safety: replace possible infs, derived from ratios (e.g, div by zero), with NaN
    if cfg.replace_infs_with_nan:
        out = _replace_infs(out)

    # Resolve column groups
    if cfg.macro_cols is None or cfg.fundamental_cols is None \
       or cfg.growth_cols is None or cfg.margin_cols is None:
        raise ValueError("Column groups must be explicitly provided in PreprocessConfig.")

    macro_cols = list(cfg.macro_cols)
    fundamental_cols = list(cfg.fundamental_cols)
    growth_cols = list(cfg.growth_cols)
    margin_cols = list(cfg.margin_cols)
    

    # 1) Macro forward-fill (post effective_date already ensured upstream)
    if cfg.macro_ffill and macro_cols:
        out = _ffill_cols_within_ticker(out, cfg.ticker_col, macro_cols)

    # 2) Fundamentals forward-fill (post effective_date already ensured upstream)
    if cfg.fundamentals_ffill and fundamental_cols:
        # Preserve NaNs caused by sanity rules (e.g., equity/interest issues)
        no_ffill = {"roe", "debt_to_equity", "interest_coverage"}
        ffill_cols = [c for c in fundamental_cols if c not in no_ffill]
        if ffill_cols:
            out = _ffill_cols_within_ticker(out, cfg.ticker_col, ffill_cols)

    # 3) Domain-aware clipping (recommended)
    if cfg.domain_clip:
        dc = cfg.domain_clip_config

        # growth
        if growth_cols:
            out = _clip_cols(out, growth_cols, *dc.growth_bounds)

        # margins
        if margin_cols:
            out = _clip_cols(out, margin_cols, *dc.margin_bounds)

        # roe/roa
        if "roe" in out.columns:
            out["roe"] = out["roe"].clip(*dc.roe_bounds)
        if "roa" in out.columns:
            out["roa"] = out["roa"].clip(*dc.roa_bounds)
        if "delta_roe_yoy" in out.columns:
            out["delta_roe_yoy"] = out["delta_roe_yoy"].clip(*dc.delta_roe_bounds)

        # leverage / liquidity (optional)
        if dc.debt_to_equity_bounds is not None and "debt_to_equity" in out.columns:
            out["debt_to_equity"] = out["debt_to_equity"].clip(*dc.debt_to_equity_bounds)
        if dc.interest_coverage_bounds is not None and "interest_coverage" in out.columns:
            out["interest_coverage"] = out["interest_coverage"].clip(*dc.interest_coverage_bounds)
        if dc.asset_turnover_bounds is not None and "asset_turnover" in out.columns:
            out["asset_turnover"] = out["asset_turnover"].clip(*dc.asset_turnover_bounds)
        if dc.fcf_assets_bounds is not None and "fcf_assets" in out.columns:
            out["fcf_assets"] = out["fcf_assets"].clip(*dc.fcf_assets_bounds)

    # 4) Column-level NaN pruning (disabled by default)
    if cfg.prune_sparse_columns:
        out = _prune_sparse_columns(out, cfg.min_non_na_ratio)

    # Final sort/reset
    out = out.sort_values([cfg.date_col, cfg.ticker_col]).reset_index(drop=True)
    logger.info("Preprocess panel complete: %d rows, %d columns", out.shape[0], out.shape[1])
    return out
