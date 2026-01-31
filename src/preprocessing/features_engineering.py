import numpy as np
import pandas as pd


# Helpers
def sign_with_tol(x: pd.Series, tol: float = 1e-12) -> pd.Series:
    """Return -1/0/+1 with a tolerance around zero."""
    x = x.astype("float64")
    out = pd.Series(np.where(x > tol, 1, np.where(x < -tol, -1, 0)), index=x.index)
    return out.astype("int8")


def fill_zeros_with_last_nonzero(s: pd.Series) -> pd.Series:
    """
    Replace 0 with last non-zero value (forward fill), keeps initial zeros if no prior sign.
    Useful to avoid 'stable' states breaking the 4-regime mapping.
    """
    s2 = s.copy()
    s2 = s2.replace(0, np.nan).ffill().fillna(0).astype("int8")
    return s2


def tercile_bucket_cs(df: pd.DataFrame, date_col: str, col: str, labels=("low", "mid", "high")) -> pd.Series:
    """
    Cross-sectional tercile buckets per date using percentile rank.
    Robust with small N (like 10 tickers) and avoids qcut duplicates issues.
    """
    pct = df.groupby(date_col)[col].rank(pct=True, method="average")
    # pct in (0,1], map to 0/1/2
    b = pd.cut(
        pct,
        bins=[0.0, 1/3, 2/3, 1.0],
        labels=list(labels),
        include_lowest=True,
    )
    return b.astype("category")


def macro_regime_from_signs(cpi_trend: pd.Series, gdp_trend: pd.Series) -> pd.Series:
    """
    Map CPI trend sign and GDP trend sign to 4 macro regimes.
    Assumes signs are in {-1, 0, +1}. We'll treat 0 as "flat" and keep it as its own label 'unclear'
    or, more simply, pre-fill zeros before calling this function.
    """
    # Expecting already filled zeros -> last non-zero
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


# Main feature engineering function
def engineer_features(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, ticker_col]).reset_index(drop=True)

    # ================
    # 1) MACRO TRENDS
    # ================
    # Using trading-day approximations:
    # ~12 months ≈ 252 trading days
    # ~6 months  ≈ 126 trading days
    # GDP 4 quarters ≈ 252 trading days (GDP changes quarterly but it's ok on daily panel)

    # Deltas (level changes)
    df["cpi_delta_12m"] = df["CPIAUCSL"] - df.groupby(ticker_col)["CPIAUCSL"].shift(252)
    df["rate_delta_6m"] = df["FEDFUNDS"] - df.groupby(ticker_col)["FEDFUNDS"].shift(126)
    df["gdp_delta_4q"] = df["GDP"] - df.groupby(ticker_col)["GDP"].shift(252)

    df["cpi_trend"] = sign_with_tol(df["cpi_delta_12m"])
    df["rate_trend"] = sign_with_tol(df["rate_delta_6m"])
    df["gdp_trend"] = sign_with_tol(df["gdp_delta_4q"])

    # Optional but recommended: replace 0 (flat) with last non-zero to keep regimes consistent
    df["cpi_trend"] = fill_zeros_with_last_nonzero(df["cpi_trend"])
    df["gdp_trend"] = fill_zeros_with_last_nonzero(df["gdp_trend"])
    df["rate_trend"] = fill_zeros_with_last_nonzero(df["rate_trend"])

    df["macro_regime"] = macro_regime_from_signs(df["cpi_trend"], df["gdp_trend"])

    # Drop helper deltas if you want (you can keep them too; here we drop to stay clean)
    df.drop(columns=["cpi_delta_12m", "rate_delta_6m", "gdp_delta_4q"], inplace=True)

    # ==========================
    # 2) FUNDAMENTALS (CS buckets)
    # ==========================
    # Buckets per date: low/mid/high
    df["margin_bucket"] = tercile_bucket_cs(df, date_col, "net_margin")
    df["profitability_bucket"] = tercile_bucket_cs(df, date_col, "roe")
    df["leverage_bucket"] = tercile_bucket_cs(df, date_col, "debt_to_equity")

    # Flags
    df["is_profitable"] = (df["net_margin"] > 0).astype("int8")

    # Cross-sectional medians
    rev_growth_med = df.groupby(date_col)["revenue_growth_qoq"].transform("median")
    dte_med = df.groupby(date_col)["debt_to_equity"].transform("median")

    df["is_high_growth"] = (df["revenue_growth_qoq"] > rev_growth_med).astype("int8")
    df["is_high_leverage"] = (df["debt_to_equity"] > dte_med).astype("int8")

    # ======================
    # 3) TECHNICAL (states)
    # ======================
    # RSI state: -1 oversold, 0 neutral, +1 overbought
    rsi = df["rsi_14"].astype("float64")
    df["rsi_state"] = pd.Series(
        np.where(rsi < 30, -1, np.where(rsi > 70, 1, 0)),
        index=df.index
    ).astype("int8")

    # Trend state: bull if adj_close > sma_200, else bear
    # If sma_200 is NaN (warm-up), keep NA (do not force 0/1)
    trend = df["adj_close"] > df["sma_200"]
    df["trend_state"] = pd.Series(np.where(df["sma_200"].isna(), pd.NA, trend.astype(int)), index=df.index).astype("Int64")

    return df