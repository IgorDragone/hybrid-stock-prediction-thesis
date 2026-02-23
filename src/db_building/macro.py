# src/data/macro.py
"""Module to fetch macroeconomic data from FRED with caching support."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas_datareader import data as web


def _macro_cache_path(
    series: list[str],
    start: str,
    end: str,
    cache_dir: Path | str | None,
) -> Path | None:
    """
    Cache file name that depends on series + date range.

    Args:
        series (list[str]): List of FRED series ids
        start (str): Start date string
        end (str): End date string
        cache_dir (Path | str | None): Directory to store cache files
    
    Returns:
        Path | None: Path to cache file or None if cache_dir is None
    """
    if cache_dir is None:
        return None
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    series_tag = "_".join(series)
    start_year = start.split("-")[0]
    end_year = end.split("-")[0]

    return cache_dir / f"fred_{series_tag}_{start_year}_{end_year}.parquet"


def fetch_macro_fred(
    series: list[str],
    start: str,
    end: str,
    cache_dir: Path | str | None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED for given series between start and end dates.
    
    Args:
        series (list[str]): List of FRED series ids
        start (str): Start date string in 'YYYY-MM-DD' format
        end (str): End date string in 'YYYY-MM-DD' format
        cache_dir (Path | str | None): Directory to store cache files
        use_cache (bool): Whether to use cached data if available
        force_refresh (bool): Whether to ignore cache and fetch fresh data
    
    Returns:
        pd.DataFrame: DataFrame with macroeconomic data indexed by date. Columns correspond to series ids.
    """
    cache_path = _macro_cache_path(series, start, end, cache_dir)

    # Check cache
    if use_cache and cache_path is not None and cache_path.exists() and not force_refresh:
        macro = pd.read_parquet(cache_path)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.sort_index()
        return macro

    # Fetch from FRED
    frames = []
    for s in series:
        ser = web.DataReader(s, "fred", start, end)
        ser.columns = [s]
        frames.append(ser)

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    # Save to cache
    if use_cache and cache_path is not None:
        macro.to_parquet(cache_path)

    return macro


def align_macro_monthly(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Align macro series to monthly frequency (end of month) and derive YoY growth.
    """
    m = macro.copy()
    m.index = pd.to_datetime(m.index)
    m = m.sort_index()
    m = m.resample("M").last()

    if "CPIAUCSL" in m.columns:
        m["cpi_yoy"] = m["CPIAUCSL"].pct_change(12)
    if "INDPRO" in m.columns:
        m["ip_yoy"] = m["INDPRO"].pct_change(12)

    return m
