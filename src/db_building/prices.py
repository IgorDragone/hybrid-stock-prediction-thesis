# src/data/prices.py
"""Module to fetch daily OHLCV prices from Yahoo Finance with caching support."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def _prices_cache_path(ticker: str, cache_dir: Path | str | None) -> Path | None:
    """
    Get the Parquet cache path for a ticker's prices. Ensures the directory exists, creating it if necessary.

    Args:
        ticker (str): Stock ticker symbol
        cache_dir (Path | str | None): Directory to store cached Parquet files

    Returns:
        Path | None: The path to the Parquet cache file for the ticker or None if cache_dir is None
    """

    if cache_dir is None:
        return None
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{ticker}.parquet"


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV prices for a ticker from Yahoo Finance, with Parquet caching.

    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format
        cache_dir (Path | str | None): Directory to store cached Parquet files
        use_cache (bool): Whether to use cached data if available
        force_refresh (bool): Whether to ignore cache and fetch fresh data

    Returns:
        pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'adj_close', 'volume'] indexed by date
    """
    cache_path = _prices_cache_path(ticker, cache_dir)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Check cache
    if use_cache and cache_path is not None and cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        if df.index.min() <= start_ts and df.index.max() >= end_ts:
            return df.loc[start:end]

    # Fetch from Yahoo Finance
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No price data for {ticker}")

    # Flatten MultiIndex (yf returns it even for 1 ticker sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Cache full series (not only [start:end]) so future builds are faster
    if use_cache and cache_path is not None:
        df.to_parquet(cache_path)

    return df
