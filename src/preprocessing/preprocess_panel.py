# src/preprocessing/preprocess_panel.py
"""
Preprocessing utilities for a daily long-format panel dataset.

The module applies domain-aware stabilization (winsorization/clipping),
macro forward-fill, and feature transformations while preserving
panel integrity (sorted, unique (date, ticker)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd

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
        current_ratio_bounds (Optional[Tuple[float, float]]): min/max bounds for current ratio
        extra_bounds (Dict[str, Tuple[float, float]]): additional column-specific bounds
    """
    # Growth rates: asymmetrically bounded (base effects)
    growth_bounds: Tuple[float, float] = (-0.5, 2.0)  # [-50%, +200%]

    # Margins: economically bounded-ish. Allow negatives but clip extreme tails.
    margin_bounds: Tuple[float, float] = (-1.0, 1.0)  # [-100%, +100%]

    # ROE/ROA can spike when equity/assets are small/negative. Clip to reasonable ranges.
    roe_bounds: Tuple[float, float] = (-2.0, 2.0)
    roa_bounds: Tuple[float, float] = (-1.0, 1.0)

    # Optional: leverage/ratios can be huge when denominator tiny; clip if desired.
    # Set to None to disable.
    debt_to_equity_bounds: Optional[Tuple[float, float]] = (0.0, 10.0)
    current_ratio_bounds: Optional[Tuple[float, float]] = (0.0, 10.0)

    # In case other columns need clipping
    extra_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for panel preprocessing.

    Attributes:
        macro_ffill (bool): Whether to forward-fill macro variables.
        winsorize_fundamentals_cs (bool): Apply cross-sectional winsorization.
        domain_clip (bool): Enable domain-aware clipping of ratios.
        add_log_volume (bool): Add log-transformed trading volume.
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

    # Fundamental stabilization
    winsorize_fundamentals_cs: bool = True
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99

    # Domain clipping
    domain_clip: bool = True
    domain_clip_config: DomainClipConfig = DomainClipConfig()

    # Transformations
    add_log_volume: bool = True
    volume_col: str = "volume"
    log_volume_col: str = "log_volume"

    # Column-level NaN pruning
    min_non_na_ratio: float = 0.95

    # Safety
    enforce_sorted: bool = True
    assert_unique_panel: bool = True


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
    df: pd.DataFrame, date_col: str, ticker_col: str, cols: Sequence[str]
) -> pd.DataFrame:
    """
    Forward-fill selected columns within each ticker over time.

    This is used to propagate the most recent available value (e.g., macro
    features) to subsequent days without leaking information across tickers.
    """
    if not cols:
        return df
    
    out = df.copy()
    # Sort so ffill respects temporal order within each ticker
    out = out.sort_values([date_col, ticker_col])
    out[list(cols)] = out.groupby(ticker_col, group_keys=False)[list(cols)].ffill()

    return out


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

    def _clip_group(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        for c in cols:
            s = gg[c]
            # Avoid unstable quantiles when too few observations exist for that date
            if s.notna().sum() < min_points:
                continue
            lo = s.quantile(lower_q)
            hi = s.quantile(upper_q)
            gg[c] = s.clip(lo, hi)
        return gg

    out = out.groupby(date_col, group_keys=False).apply(_clip_group)
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


#-----------------------------------------------
# Main preprocessing function
#-----------------------------------------------
def preprocess_panel(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
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
        6) Cross-sectional winsorization of fundamental ratios (per date).
        7) Domain-aware clipping of selected ratios (optional).
        8) Add log-volume feature (optional).
        9) Prune columns with excessive missing values.

    Args:
        df (pd.DataFrame): Long-format panel with at least (date, ticker).
        config (PreprocessConfig): Preprocessing configuration.

    Returns:
        pd.DataFrame: Processed panel sorted by (date, ticker) with a fresh integer index.
    """
    _check_required_columns(df, [config.date_col, config.ticker_col])

    out = df.copy()
    out = _ensure_datetime(out, config.date_col)

    # As we need to forward-fill and do cross-sectional ops, ensure sortedness
    if config.enforce_sorted:
        out = out.sort_values([config.date_col, config.ticker_col])

    # Very quick uniqueness check, as DB build should guarantee this
    if config.assert_unique_panel:
        _assert_unique_panel(out, config.date_col, config.ticker_col)

    # Safety: replace possible infs, derived from ratios (e.g, div by zero), with NaN
    if config.replace_infs_with_nan:
        out = _replace_infs(out)

    # Resolve column groups
    if config.macro_cols is None or config.fundamental_cols is None \
       or config.growth_cols is None or config.margin_cols is None:
        raise ValueError("Column groups must be explicitly provided in PreprocessConfig.")

    macro_cols = list(config.macro_cols)
    fundamental_cols = list(config.fundamental_cols)
    growth_cols = list(config.growth_cols)
    margin_cols = list(config.margin_cols)
    

    # 1) Macro forward-fill (post effective_date already ensured upstream)
    if config.macro_ffill and macro_cols:
        out = _ffill_cols_within_ticker(out, config.date_col, config.ticker_col, macro_cols)

    # 2) Cross-sectional winsorization for fundamentals (stabilizes ratios)
    if config.winsorize_fundamentals_cs and fundamental_cols:
        out = _winsorize_cross_sectional(
            out, date_col=config.date_col, cols=fundamental_cols,
            lower_q=config.winsor_lower_q, upper_q=config.winsor_upper_q
        )

    # 3) Domain-aware clipping (recommended)
    if config.domain_clip:
        dc = config.domain_clip_config

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

        # leverage / liquidity (optional)
        if dc.debt_to_equity_bounds is not None and "debt_to_equity" in out.columns:
            out["debt_to_equity"] = out["debt_to_equity"].clip(*dc.debt_to_equity_bounds)
        if dc.current_ratio_bounds is not None and "current_ratio" in out.columns:
            out["current_ratio"] = out["current_ratio"].clip(*dc.current_ratio_bounds)

        # extras
        for col, (lo, hi) in dc.extra_bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lo, hi)

    # 4) Add log(volume)
    if config.add_log_volume and config.volume_col in out.columns:
        out[config.log_volume_col] = np.log1p(out[config.volume_col])

    # 5) Column-level NaN pruning (avoid dropping rows at this stage)
    out = _prune_sparse_columns(out, config.min_non_na_ratio)

    # Final sort/reset
    out = out.sort_values([config.date_col, config.ticker_col]).reset_index(drop=True)
    
    return out