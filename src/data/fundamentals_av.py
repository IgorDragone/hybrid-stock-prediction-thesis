# src/data/fundamentals_av.py
"""Fetch quarterly fundamentals from Alpha Vantage API with caching."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


AV_URL = "https://www.alphavantage.co/query"


def _av_request(function: str, symbol: str, api_key: str, timeout: int = 30) -> dict:
    """
    Make a request to the Alpha Vantage API.

    Args:
        function(str): Alpha Vantage function name
        symbol(str): Stock ticker symbol
        api_key(str): Alpha Vantage API key
        timeout(int): Request timeout in seconds
    
    Returns:
        dict: Parsed JSON response as a dictionary.
    """
    params = {"function": function, "symbol": symbol, "apikey": api_key}
    r = requests.get(AV_URL, params=params, timeout=timeout)
    r.raise_for_status()

    return r.json()


def _load_or_fetch_json(
    cache_path: Path,
    function: str,
    symbol: str,
    api_key: str,
    sleep_seconds: float = 12.5,  # ~5 req/min safe
) -> dict:
    """
    Load JSON data from cache or fetch from Alpha Vantage API if not cached.

    Args:
        cache_path(Path): Path to cache file
        function(str): Alpha Vantage function name
        symbol(str): Stock ticker symbol
        api_key(str): Alpha Vantage API key
        sleep_seconds(float): Seconds to sleep after fetching to respect rate limits
    
    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data = _av_request(function=function, symbol=symbol, api_key=api_key)

    # HARD FAILS: DO NOT CACHE THESE RESPONSES (something went wrong)
    if "Information" in data:
        # daily rate limit message
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Information'][:200]}")

    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage note: {data['Note'][:200]}")

    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")

    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("Alpha Vantage returned empty response.")

    # SAFE TO CACHE
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    time.sleep(sleep_seconds)

    return data


def _to_num(s: pd.Series) -> pd.Series:
    # AV often returns numeric values as strings, sometimes "None"
    return pd.to_numeric(s.replace({"None": None, "": None}), errors="coerce")


def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    # Safely get a numeric column from DataFrame, or return a Series of NaNs if missing
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    return _to_num(df[col])


def fetch_quarterly_fundamentals_av(
    ticker: str,
    api_key: str,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals from Alpha Vantage and return a reduced quarterly wide DataFrame.

    Args:
        ticker (str): Stock ticker symbol
        api_key (str): Alpha Vantage API key
        cache_dir (str | Path): Directory to cache JSON responses
    
    Returns:
        pd.DataFrame: Quarterly fundamentals DataFrame with the following output columns:
            - fiscalDateEnding (datetime)
            - net_margin, operating_margin, ebitda_margin
            - asset_turnover, roe, roa
            - fcf_margin
            - revenue_growth_qoq, earnings_growth_qoq, fcf_growth_qoq
            - debt_to_equity, current_ratio

    Notes:
      - No lag applied here. Apply lag later with effective_date + merge_asof.
    """
    cache_dir = Path(cache_dir)

    income_json = _load_or_fetch_json(
        cache_path=cache_dir / "income" / f"{ticker}.json",
        function="INCOME_STATEMENT",
        symbol=ticker,
        api_key=api_key,
    )
    balance_json = _load_or_fetch_json(
        cache_path=cache_dir / "balance" / f"{ticker}.json",
        function="BALANCE_SHEET",
        symbol=ticker,
        api_key=api_key,
    )
    cashflow_json = _load_or_fetch_json(
        cache_path=cache_dir / "cashflow" / f"{ticker}.json",
        function="CASH_FLOW",
        symbol=ticker,
        api_key=api_key,
    )

    inc = pd.DataFrame(income_json.get("quarterlyReports", []))
    bal = pd.DataFrame(balance_json.get("quarterlyReports", []))
    cf = pd.DataFrame(cashflow_json.get("quarterlyReports", []))

    if inc.empty or bal.empty:
        # cash flow can be empty for some tickers; income+balance is the minimum
        raise ValueError(f"No quarterly fundamentals from AV for {ticker}")

    # Standardize date
    for df in (inc, bal, cf):
        if not df.empty and "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")

    # Merge statements on fiscalDateEnding
    q = inc.merge(bal, on="fiscalDateEnding", how="outer", suffixes=("", "_bal"))
    if not cf.empty:
        q = q.merge(cf, on="fiscalDateEnding", how="outer", suffixes=("", "_cf"))

    q = q.sort_values("fiscalDateEnding").reset_index(drop=True)

    # Core line items 
    revenue = _safe_get(q, "totalRevenue")
    net_income = _safe_get(q, "netIncome")
    operating_income = _safe_get(q, "operatingIncome")
    ebitda = _safe_get(q, "ebitda")

    total_assets = _safe_get(q, "totalAssets")
    total_equity = _safe_get(q, "totalShareholderEquity")

    # debt field can vary; AV often exposes shortLongTermDebtTotal
    total_debt = _safe_get(q, "shortLongTermDebtTotal")
    if total_debt.isna().all():
        # fallback: totalLiabilities (less ideal, but better than nothing)
        total_debt = _safe_get(q, "totalLiabilities")

    current_assets = _safe_get(q, "totalCurrentAssets")
    current_liabilities = _safe_get(q, "totalCurrentLiabilities")

    # Cash flow: FCF = operating cash flow - capex (capex is typically negative in many feeds; handle both cases)
    op_cf = _safe_get(q, "operatingCashflow")
    capex = _safe_get(q, "capitalExpenditures")
    # If capex is already negative, op_cf - capex increases; if positive, op_cf - capex reduces. This formula is standard.
    fcf = op_cf - capex

    out = pd.DataFrame({"fiscalDateEnding": q["fiscalDateEnding"]})

    # Profitability
    out["net_margin"] = net_income / revenue
    out["operating_margin"] = operating_income / revenue
    out["ebitda_margin"] = ebitda / revenue
    out["asset_turnover"] = revenue / total_assets
    out["roe"] = net_income / total_equity
    out["roa"] = net_income / total_assets

    # Cash quality / margins
    out["fcf_margin"] = fcf / revenue

    # Growth (QoQ)
    out["revenue_growth_qoq"] = revenue.pct_change(1)
    out["earnings_growth_qoq"] = net_income.pct_change(1)
    out["fcf_growth_qoq"] = fcf.pct_change(1)

    # Risk / liquidity
    out["debt_to_equity"] = total_debt / total_equity
    out["current_ratio"] = current_assets / current_liabilities

    # Clean
    out = out.dropna(subset=["fiscalDateEnding"]).sort_values("fiscalDateEnding")

    return out.reset_index(drop=True)
