"""
Feature engineering for daily financial panel data.

This module derives:
- Macro trend signals and a 4-regime macro state
- Cross-sectional (per-date) percentile-rank features for fundamentals/technicals
  computed on end-of-month snapshots to align with monthly rebalancing
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
def _cs_percentile_or_zscore(
    df: pd.DataFrame,
    date_col: str,
    col: str,
    min_points: int = 15,
) -> pd.Series:
    """
    Cross-sectional percentile rank per date with a fallback to z-score when
    the cross section is too small.
    """
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")

    def _transform(group_df: pd.DataFrame) -> pd.Series:
        series = group_df[col].astype("float64")
        valid_count = series.notna().sum()
        if valid_count < min_points:
            std_dev = series.std(ddof=0)
            if std_dev == 0 or pd.isna(std_dev):
                return pd.Series([0.0] * len(series), index=series.index)
            return (series - series.mean()) / std_dev
        return series.rank(pct=True, method="average")

    return df.groupby(date_col, group_keys=False).apply(_transform)


# -----------------------------------------------
# Main feature engineering function
# -----------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Engineer macro, fundamental, and technical features for an EOM panel dataset.

    Expects EOM (end-of-month) snapshots as input.

    Steps:
        1) Macro trends on EOM:
            - Derive a 4-regime macro state
        2) Fundamentals on EOM:
            - Cross-sectional percentile-rank features
        3) Technicals on EOM:
            - Cross-sectional percentile-rank features

    Args:
        df (pd.DataFrame): Input DataFrame with necessary columns.
        date_col (str): Name of the date column.
        ticker_col (str): Name of the ticker/asset identifier column.
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

    df_eom = df.copy()

    # 1) MACRO TRENDS (EOM)
    # Monthly macro features (aligned to end-of-month in DB build)
    # cpi_yoy/ip_yoy are derived in db_building.macro and merged into the daily panel.
    df_eom["slope_10y2y"] = df_eom["DGS10"] - df_eom["DGS2"]
    df_eom["curve_inverted"] = (df_eom["slope_10y2y"] < 0).astype("int8")

    df_eom["stress_index"] = df_eom["STLFSI4"]
    df_eom["risk_off_flag"] = (df_eom["stress_index"] > 0.5).astype("int8")

    # Growth–inflation regimes (analysis only)
    df_eom["growth_up"] = df_eom["ip_yoy"] > 0
    cpi_roll = df_eom.groupby(ticker_col)["cpi_yoy"].transform(
        lambda s: s.rolling(60, min_periods=12).mean(),
    )
    df_eom["inflation_up"] = df_eom["cpi_yoy"] > cpi_roll

    df_eom["macro_regime"] = np.select(
        [
            df_eom["growth_up"] & (~df_eom["inflation_up"]),
            df_eom["growth_up"] & df_eom["inflation_up"],
            (~df_eom["growth_up"]) & df_eom["inflation_up"],
            (~df_eom["growth_up"]) & (~df_eom["inflation_up"]),
        ],
        ["goldilocks", "reflation", "stagflation", "deflation"],
        default="unknown",
    ).astype("category")
    df_eom["macro_regime_label"] = df_eom["macro_regime"]

    # 2) FUNDAMENTALS (EOM)
    # Fundamental percentile-rank features (model-ready)
    if "roe" in df_eom.columns:
        df_eom["roe_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "roe").fillna(0.5)
    if "roa" in df_eom.columns:
        df_eom["roa_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "roa").fillna(0.5)
    if "operating_margin" in df_eom.columns:
        df_eom["operating_margin_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "operating_margin").fillna(0.5)
    if "gross_margin" in df_eom.columns:
        df_eom["gross_margin_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "gross_margin").fillna(0.5)
    if "revenue_growth_yoy" in df_eom.columns:
        df_eom["revenue_growth_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "revenue_growth_yoy").fillna(0.5)
    if "earnings_growth_yoy" in df_eom.columns:
        df_eom["earnings_growth_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "earnings_growth_yoy").fillna(0.5)
    if "delta_roe_yoy" in df_eom.columns:
        df_eom["delta_roe_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "delta_roe_yoy").fillna(0.5)
    if "debt_to_equity" in df_eom.columns:
        df_eom["debt_pr"] = _cs_percentile_or_zscore(
            df_eom.assign(_tmp=-df_eom["debt_to_equity"]),
            date_col,
            "_tmp",
        ).fillna(0.5)
    if "interest_coverage" in df_eom.columns:
        df_eom["interest_coverage_pr"] = _cs_percentile_or_zscore(
            df_eom,
            date_col,
            "interest_coverage",
        ).fillna(0.5)
    if "asset_turnover" in df_eom.columns:
        df_eom["asset_turnover_pr"] = _cs_percentile_or_zscore(
            df_eom,
            date_col,
            "asset_turnover",
        ).fillna(0.5)

    # 3) TECHNICAL percentile-rank features (model-ready)

    if "volatility_20d" in df_eom.columns and "volatility_60d" in df_eom.columns:
        df_eom["vol_ratio"] = df_eom["volatility_20d"] / df_eom["volatility_60d"]

    if "mom_12m" in df_eom.columns:
        df_eom["mom12_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "mom_12m")
    if "mom_6m" in df_eom.columns:
        df_eom["mom6_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "mom_6m")
    if "mom_3m" in df_eom.columns:
        df_eom["mom3_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "mom_3m")
    if "price_sma_200" in df_eom.columns:
        df_eom["trend_ratio_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "price_sma_200")
    if "volatility_60d" in df_eom.columns:
        df_eom["vol_pr"] = _cs_percentile_or_zscore(df_eom.assign(_tmp=-df_eom["volatility_60d"]), date_col, "_tmp")
    if "vol_ratio" in df_eom.columns:
        df_eom["vol_ratio_pr"] = _cs_percentile_or_zscore(df_eom, date_col, "vol_ratio")

    tech_pr_cols = [
        "mom12_pr",
        "mom6_pr",
        "mom3_pr",
        "trend_ratio_pr",
        "vol_pr",
        "vol_ratio_pr",
    ]
    tech_pr_cols = [c for c in tech_pr_cols if c in df_eom.columns]
    for col in tech_pr_cols:
        df_eom[col] = df_eom[col].fillna(0.5)

    macro_cols = [
        "slope_10y2y",
        "curve_inverted",
        "stress_index",
        "risk_off_flag",
        "growth_up",
        "inflation_up",
        "macro_regime",
        "macro_regime_label",
    ]
    macro_cols = [c for c in macro_cols if c in df_eom.columns]

    fund_pr_cols = [
        "roe_pr",
        "roa_pr",
        "operating_margin_pr",
        "gross_margin_pr",
        "revenue_growth_pr",
        "earnings_growth_pr",
        "delta_roe_pr",
        "debt_pr",
        "interest_coverage_pr",
        "asset_turnover_pr",
    ]
    fund_pr_cols = [c for c in fund_pr_cols if c in df_eom.columns]

    logger.info("Feature engineering complete: %d rows, %d columns", df_eom.shape[0], df_eom.shape[1])
    return df_eom
