"""
Feature engineering for daily financial panel data.

This module derives:
- Macro trend signals and a 4-regime macro state
- Cross-sectional (per-date) fundamental buckets and binary flags
- Simple technical indicator states (RSI, trend vs SMA)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# -----------------------------------------------
# Helpers functions
# -----------------------------------------------
def _sign_with_tolerance(x: pd.Series, tol: float = 1e-12) -> pd.Series:
    # -Convert a numeric series to {-1, 0, +1}, using a tolerance around zero
    x = x.astype("float64")
    out = pd.Series(np.where(x > tol, 1, np.where(x < -tol, -1, 0)), index=x.index)
    return out.astype("int8")


def _fill_zeros_with_last_nonzero(s: pd.Series) -> pd.Series:
    """
    Replace 0 with last non-zero value (forward fill), keeps initial zeros if no prior sign.
    Useful to avoid 'stable' states breaking the 4-regime mapping.
    """
    s2 = s.copy()
    s2 = s2.replace(0, np.nan).ffill().fillna(0).astype("int8")
    return s2


def _tercile_bucket_cs(df: pd.DataFrame, date_col: str, col: str, labels=("low", "mid", "high")) -> pd.Series:
    # Assign cross-sectional tercile buckets per date based on percentile rank.
    pct = df.groupby(date_col)[col].rank(pct=True, method="average") # pct in (0,1], map to 0/1/2
    buckets = pd.cut(
        pct,
        bins=[0.0, 1/3, 2/3, 1.0],
        labels=list(labels),
        include_lowest=True,
    )
    return buckets.astype("category")


def _first_available_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _macro_regime_from_signs(cpi_trend: pd.Series, gdp_trend: pd.Series) -> pd.Series:
    """
    Map CPI/GDP trend signs to four macro regimes.

    Expected inputs are sign-coded series in {-1, 0, +1}. In this project we
    typically replace 0 (flat) with the last non-zero sign prior to calling,
    to avoid frequent regime switching due to small numerical fluctuations.
    """
    up_cpi = cpi_trend > 0
    up_gdp = gdp_trend > 0

    regime = np.where(
        up_cpi & up_gdp, "expansion",
        np.where(
            up_cpi & (~up_gdp), "stagflation",
            np.where(
                (~up_cpi) & up_gdp, "disinflation",
                "recession"
            )
        )
    )
    return pd.Series(regime, index=cpi_trend.index).astype("category")


# -----------------------------------------------
# Main feature engineering function
# -----------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Engineer macro, fundamental, and technical features for a daily panel dataset.

    Steps:
        1) Macro trends:
            - Compute multi-month deltas (trading-day approximations)
            - Convert deltas to sign trends and derive a 4-regime macro state
        2) Fundamentals:
            - Create cross-sectional tercile buckets per date
            - Create binary flags relative to cross-sectional medians
        3) Technical states:
            - RSI state (oversold/neutral/overbought)
            - Trend state vs SMA(200), preserving NA during indicator warm-up

    Args:
        df (pd.DataFrame): Input DataFrame with necessary columns.
        date_col (str): Name of the date column.
        ticker_col (str): Name of the ticker/asset identifier column.

    Returns:
        pd.DataFrame: Input DataFrame enriched with engineered features.
    """
    logger.info("Engineer features: %d rows, %d columns", df.shape[0], df.shape[1])
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, ticker_col]).reset_index(drop=True)

    # 1) MACRO TRENDS
    # Monthly macro features (aligned to end-of-month in DB build)
    df["cpi_yoy"] = df.groupby(ticker_col)["CPIAUCSL"].pct_change(12)
    df["ip_yoy"] = df.groupby(ticker_col)["INDPRO"].pct_change(12)

    df["slope_10y2y"] = df["DGS10"] - df["DGS2"]
    df["curve_inverted"] = (df["slope_10y2y"] < 0).astype("int8")

    df["stress_level"] = df["STLFSI4"]
    df["risk_off"] = (df["stress_level"] > 0.5).astype("int8")

    # Growth–inflation regimes (analysis only)
    df["growth_up"] = df["ip_yoy"] > 0
    cpi_roll = df.groupby(ticker_col)["cpi_yoy"].transform(lambda s: s.rolling(60, min_periods=12).mean())
    df["inflation_up"] = df["cpi_yoy"] > cpi_roll

    df["macro_regime"] = np.select(
        [
            df["growth_up"] & (~df["inflation_up"]),
            df["growth_up"] & df["inflation_up"],
            (~df["growth_up"]) & df["inflation_up"],
            (~df["growth_up"]) & (~df["inflation_up"]),
        ],
        ["goldilocks", "reflation", "stagflation", "deflation"],
        default="unknown",
    ).astype("category")

    # 2) FUNDAMENTALS (CS buckets)
    
    # Buckets per date: low/mid/high
    margin_col = _first_available_col(df, ["operating_margin", "gross_margin", "net_margin"])
    if margin_col:
        df["margin_bucket"] = _tercile_bucket_cs(df, date_col, margin_col)
    else:
        df["margin_bucket"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    df["profitability_bucket"] = _tercile_bucket_cs(df, date_col, "roe")
    df["leverage_bucket"] = _tercile_bucket_cs(df, date_col, "debt_to_equity")

    # Flags
    if margin_col:
        df["is_profitable"] = (df[margin_col] > 0).astype("int8")
    else:
        df["is_profitable"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # Cross-sectional medians
    growth_col = _first_available_col(df, ["revenue_growth_yoy", "revenue_growth_qoq"])
    if growth_col:
        rev_growth_med = df.groupby(date_col)[growth_col].transform("median")
    else:
        rev_growth_med = pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    dte_med = df.groupby(date_col)["debt_to_equity"].transform("median")

    if growth_col:
        df["is_high_growth"] = (df[growth_col] > rev_growth_med).astype("int8")
    else:
        df["is_high_growth"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    df["is_high_leverage"] = (df["debt_to_equity"] > dte_med).astype("int8")

    # Fundamental deltas (YoY)
    if "roe" in df.columns:
        df["delta_roe_yoy"] = df.groupby(ticker_col)["roe"].transform(lambda s: s - s.shift(4))
    else:
        df["delta_roe_yoy"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")


    # 3) TECHNICAL (states)
    if "rsi_14" in df.columns:
        # RSI state: -1 oversold, 0 neutral, +1 overbought
        rsi = df["rsi_14"].astype("float64")
        df["rsi_state"] = pd.Series(
            np.where(rsi < 30, -1, np.where(rsi > 70, 1, 0)),
            index=df.index
        ).astype("int8")
    else:
        df["rsi_state"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # Trend state: bull if adj_close > sma_200, else bear
    # If sma_200 is NaN (warm-up), keep NA (do not force 0/1)
    trend = df["adj_close"] > df["sma_200"]
    df["trend_state"] = pd.Series(np.where(df["sma_200"].isna(), pd.NA, trend.astype(int)), index=df.index).astype("Int64")

    logger.info("Feature engineering complete: %d rows, %d columns", df.shape[0], df.shape[1])
    return df
