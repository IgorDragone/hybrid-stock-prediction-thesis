# src/data/build_db.py
"""Build financial database parquet from prices, technicals, fundamentals, and macro data."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.db_buiding.config import (
    FRED_SERIES,
    FUNDAMENTALS_LAG_DAYS,
    MACRO_LAG_DAYS,        
    AV_API_KEY,
    RAW_AV_DIR,
    RAW_PRICES_DIR,
    RAW_MACRO_DIR,
)

from src.db_buiding.prices import fetch_prices
from src.db_buiding.technicals import technical_indicators
from src.db_buiding.macro import fetch_macro_fred
from src.db_buiding.fundamentals_av import fetch_quarterly_fundamentals_av


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _to_daily_df(df_idx: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'date' column and sorted ascending.
    
    Args:
        df_idx (pd.DataFrame): DataFrame with DatetimeIndex or 'date' column.

    Returns:
        pd.DataFrame: DataFrame with 'date' column sorted ascending.
    """
    if isinstance(df_idx.index, pd.DatetimeIndex):
        df = df_idx.reset_index().rename(columns={"index": "date"})
    else:
        df = df_idx.copy()
        if "date" not in df.columns:
            raise ValueError("Expected DatetimeIndex or a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values("date")


def merge_fundamentals_asof(daily_df: pd.DataFrame, fundamentals_q: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Merge quarterly fundamentals into daily data applying publication lag. 
    effective_date = fiscalDateEnding + lag_days; asof backward merge to daily dates.

    Args:
        daily_df (pd.DataFrame): DataFrame with 'date' column.
        fundamentals_q (pd.DataFrame): DataFrame with 'fiscalDateEnding' column.
        lag_days (int): number of days to lag fundamentals.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if fundamentals_q is None or fundamentals_q.empty:
        return daily_df

    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    f = fundamentals_q.copy()
    f["fiscalDateEnding"] = pd.to_datetime(f["fiscalDateEnding"], errors="coerce")
    f = f.dropna(subset=["fiscalDateEnding"]).sort_values("fiscalDateEnding")

    f["effective_date"] = f["fiscalDateEnding"] + pd.Timedelta(days=lag_days)
    f = f.sort_values("effective_date")

    merged = pd.merge_asof(
        d,
        f.drop(columns=["fiscalDateEnding"]),
        left_on="date",
        right_on="effective_date",
        direction="backward",
        allow_exact_matches=True,
    )

    return merged


def merge_macro_asof(
    daily_df: pd.DataFrame,
    macro: pd.DataFrame,
    macro_lag_days: dict[str, int],
) -> pd.DataFrame:
    """
    Apply publication lags to macro series using effective dates, then merge_asof to daily dates.
    Output keeps 'macro_effective_date' for sanity checks (can be dropped later).

    Args:
        daily_df (pd.DataFrame): DataFrame with 'date' column.
        macro (pd.DataFrame): DataFrame with macro series and date index or 'date' column.
        macro_lag_days (dict[str, int]): dict mapping macro series names to lag days.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    m = macro.copy()
    if isinstance(m.index, pd.DatetimeIndex):
        m = m.reset_index()
        m = m.rename(columns={m.columns[0]: "macro_date"})
    else:
        if "date" in m.columns:
            m = m.rename(columns={"date": "macro_date"})
        elif "macro_date" not in m.columns:
            raise ValueError("Macro DF must have DatetimeIndex or a date column.")

    m["macro_date"] = pd.to_datetime(m["macro_date"], errors="coerce")
    m = m.dropna(subset=["macro_date"]).sort_values("macro_date")

    # wide -> long to apply per-series lag
    m_long = m.melt(id_vars=["macro_date"], var_name="series", value_name="value")
    m_long["lag_days"] = m_long["series"].map(macro_lag_days).fillna(0).astype(int)
    m_long["macro_effective_date"] = m_long["macro_date"] + pd.to_timedelta(m_long["lag_days"], unit="D")
    m_long = m_long.sort_values(["series", "macro_effective_date"])

    # back to wide at effective dates
    m_eff = (
        m_long.pivot_table(
            index="macro_effective_date",
            columns="series",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values("macro_effective_date")
    )

    merged = pd.merge_asof(
        d,
        m_eff,
        left_on="date",
        right_on="macro_effective_date",
        direction="backward",
        allow_exact_matches=True,
    )

    return merged


def build_database(
    tickers: list[str],
    start: str,
    end: str,
    out_path: Path,
    force_refresh_prices: bool = False,
    force_refresh_macro: bool = False,
) -> None:
    """
    Build financial database parquet from prices, technicals, fundamentals, and macro data.
    
    Args:
        tickers (list[str]): List of stock tickers to include.
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        out_path (Path): Output Path for the parquet file.
        force_refresh_prices (bool): If True, refetch prices ignoring cache.
        force_refresh_macro (bool): If True, refetch macro data ignoring cache.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Macro fetched once (cached)
    macro = fetch_macro_fred(
        FRED_SERIES, start, end,
        cache_dir=RAW_MACRO_DIR,
        use_cache=True,
        force_refresh=force_refresh_macro,
    )

    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            print(f"Processing {ticker}")

            # 1) Prices (cached per ticker)
            prices = fetch_prices(
                ticker, start, end,
                cache_dir=RAW_PRICES_DIR,
                use_cache=True,
                force_refresh=force_refresh_prices,
            )
            prices_daily = _to_daily_df(prices)

            # 2) Technicals from indexed prices
            tech = technical_indicators(prices)
            tech_daily = _to_daily_df(tech)

            df = pd.merge(prices_daily, tech_daily, on="date", how="left")

            # 3) Fundamentals AV (cached json per endpoint/ticker)
            if not AV_API_KEY:
                raise RuntimeError(
                    "Missing ALPHAVANTAGE_API_KEY env var. "
                    "Set it: export ALPHAVANTAGE_API_KEY='...'"
                )

            fundamentals_q = fetch_quarterly_fundamentals_av(
                ticker=ticker,
                api_key=AV_API_KEY,
                cache_dir=RAW_AV_DIR,
            )
            df = merge_fundamentals_asof(df, fundamentals_q, FUNDAMENTALS_LAG_DAYS)

            # 4) Macro asof
            df = merge_macro_asof(df, macro, MACRO_LAG_DAYS)

            df["ticker"] = ticker
            all_frames.append(df)

        except Exception:
            logger.exception("Failed %s", ticker)

    if not all_frames:
        raise RuntimeError("No tickers were successfully built.")

    panel = pd.concat(all_frames, ignore_index=True).sort_values(["date", "ticker"])
    panel.to_parquet(out_path, index=False)
    logger.info("Saved database to %s", out_path)