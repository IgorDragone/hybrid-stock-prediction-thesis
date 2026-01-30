# src/preprocessing/preprocess_panel.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd


# Config
@dataclass(frozen=True)
class DomainClipConfig:
    """
    Domain-aware clipping rules.

    Notes:
    - Bounds are in *raw units* (e.g., margins as fraction, growth as fraction).
    - These defaults are conservative and can be tightened after EDA.
    """
    # Growth rates: asymmetrically bounded (base effects)
    growth_bounds: Tuple[float, float] = (-0.5, 2.0)  # [-50%, +200%]

    # Margins: economically bounded-ish. Allow negatives but clip extreme tails.
    margin_bounds: Tuple[float, float] = (-1.0, 1.0)  # [-100%, +100%]

    # ROE/ROA can spike when equity/assets are small/negative; clip hard.
    roe_bounds: Tuple[float, float] = (-2.0, 2.0)
    roa_bounds: Tuple[float, float] = (-1.0, 1.0)

    # Optional: leverage/ratios can be huge when denominator tiny; clip if desired.
    # Set to None to disable.
    debt_to_equity_bounds: Optional[Tuple[float, float]] = (0.0, 10.0)
    current_ratio_bounds: Optional[Tuple[float, float]] = (0.0, 10.0)

    # If you want to add more domain bounds, use extra_bounds dict:
    # {"colname": (lo, hi), ...}
    extra_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessConfig:
    date_col: str = "date"
    ticker_col: str = "ticker"

    # Column groups (explicit lists are best for your dataset)
    macro_cols: Optional[Sequence[str]] = None
    fundamental_cols: Optional[Sequence[str]] = None
    growth_cols: Optional[Sequence[str]] = None
    margin_cols: Optional[Sequence[str]] = None

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

    # Feature engineering in preprocessing
    add_log_volume: bool = True
    volume_col: str = "volume"
    log_volume_col: str = "log_volume"

    # Column-level NaN pruning
    min_non_na_ratio: float = 0.95

    # Safety
    enforce_sorted: bool = True
    assert_unique_panel: bool = True


# Helpers
def _check_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
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
    dup = int(df.duplicated([date_col, ticker_col]).sum())
    if dup:
        raise ValueError(
            f"Found {dup} duplicated rows on ({date_col}, {ticker_col}). "
            "DB build should guarantee uniqueness."
        )


def _replace_infs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return out
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _ffill_cols_within_ticker(
    df: pd.DataFrame, date_col: str, ticker_col: str, cols: Sequence[str]
) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
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
    if not cols:
        return df
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1.")

    out = df.copy()

    def _clip_group(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        for c in cols:
            s = gg[c]
            if s.notna().sum() < min_points:
                continue
            lo = s.quantile(lower_q)
            hi = s.quantile(upper_q)
            gg[c] = s.clip(lo, hi)
        return gg

    out = out.groupby(date_col, group_keys=False).apply(_clip_group)
    return out


def _clip_cols(df: pd.DataFrame, cols: Sequence[str], lo: float, hi: float) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    existing = [c for c in cols if c in out.columns]
    if existing:
        out[existing] = out[existing].clip(lower=lo, upper=hi)
    return out


def _prune_sparse_columns(df: pd.DataFrame, min_non_na_ratio: float) -> pd.DataFrame:
    if not (0.0 < min_non_na_ratio <= 1.0):
        raise ValueError("min_non_na_ratio must be in (0, 1].")
    keep = df.columns[df.notna().mean() >= min_non_na_ratio].tolist()
    return df[keep].copy()


def _default_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Best-effort defaults when user doesn't pass explicit lists.
    For your parquet, explicit lists are recommended.
    """
    cols = df.columns.tolist()
    macro_candidates = [c for c in cols if c in {"CPIAUCSL", "FEDFUNDS", "GDP"}]
    growth_candidates = [c for c in cols if c.endswith("_growth_qoq")]
    margin_candidates = [c for c in cols if c in {"net_margin", "operating_margin", "ebitda_margin", "fcf_margin"}]

    # Fundamentals = "non-price, non-tech, non-macro" is ambiguous; keep it conservative.
    # We'll include common fundamental names if present.
    known_fund = [
        "net_margin", "operating_margin", "ebitda_margin", "fcf_margin",
        "asset_turnover", "roe", "roa",
        "debt_to_equity", "current_ratio",
        "revenue_growth_qoq", "earnings_growth_qoq", "fcf_growth_qoq",
    ]
    fundamental_candidates = [c for c in known_fund if c in cols]

    return {
        "macro_cols": macro_candidates,
        "growth_cols": growth_candidates,
        "margin_cols": margin_candidates,
        "fundamental_cols": fundamental_candidates,
    }


# Main
def preprocess_panel(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """
    Preprocess a daily long-format panel.

    Returns:
        pd.DataFrame: processed, long-format, sorted by (date, ticker)
    """
    _check_required_columns(df, [config.date_col, config.ticker_col])

    out = df.copy()
    out = _ensure_datetime(out, config.date_col)

    if config.enforce_sorted:
        out = out.sort_values([config.date_col, config.ticker_col])

    if config.assert_unique_panel:
        _assert_unique_panel(out, config.date_col, config.ticker_col)

    if config.replace_infs_with_nan:
        out = _replace_infs(out)

    # Resolve column groups
    defaults = _default_column_groups(out)

    macro_cols = list(config.macro_cols) if config.macro_cols is not None else defaults["macro_cols"]
    fundamental_cols = list(config.fundamental_cols) if config.fundamental_cols is not None else defaults["fundamental_cols"]
    growth_cols = list(config.growth_cols) if config.growth_cols is not None else defaults["growth_cols"]
    margin_cols = list(config.margin_cols) if config.margin_cols is not None else defaults["margin_cols"]

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
        for c, (lo, hi) in dc.extra_bounds.items():
            if c in out.columns:
                out[c] = out[c].clip(lo, hi)

    # 4) Add log(volume)
    if config.add_log_volume and config.volume_col in out.columns:
        out[config.log_volume_col] = np.log1p(out[config.volume_col])

    # 5) Column-level NaN pruning (avoid dropping rows at this stage)
    out = _prune_sparse_columns(out, config.min_non_na_ratio)

    # Final sort/reset
    out = out.sort_values([config.date_col, config.ticker_col]).reset_index(drop=True)
    return out


def sanity_report(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> dict:
    """
    Quick diagnostics after preprocessing.
    """
    _check_required_columns(df, [config.date_col, config.ticker_col])

    num = df.select_dtypes(include=[np.number])
    any_inf = bool(np.isinf(num.to_numpy()).any()) if len(num.columns) else False

    macro_cols = list(config.macro_cols) if config.macro_cols is not None else _default_column_groups(df)["macro_cols"]
    macro_any_nan = bool(df[macro_cols].isna().any().any()) if macro_cols and all(c in df.columns for c in macro_cols) else False
    macro_nan_mean = float(df[macro_cols].isna().mean().mean()) if macro_cols and all(c in df.columns for c in macro_cols) else 0.0

    return {
        "n_rows": int(len(df)),
        "n_dates": int(df[config.date_col].nunique()),
        "n_tickers": int(df[config.ticker_col].nunique()),
        "duplicates_date_ticker": int(df.duplicated([config.date_col, config.ticker_col]).sum()),
        "any_inf_remaining": any_inf,
        "macro_any_nan": macro_any_nan,
        "macro_nan_rate_mean": macro_nan_mean,
        "non_na_ratio_min": float(df.notna().mean().min()),
    }
