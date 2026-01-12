# src/data/fundamentals.py

import yfinance as yf
import pandas as pd
import numpy as np

def safe_col(df: pd.DataFrame, candidates: list) -> pd.Series | None:
    """
    Safely select a column from a DataFrame, trying multiple candidate names.

    Args:
        df (pd.DataFrame): The DataFrame to select from.
        candidates (list): A list of candidate column names to try.

    Returns:
        pd.Series: The selected column or None if no match is found.
    """
    for col in candidates:
        if col in df.columns:
            return df[col]
    print(f"Warning: None of the candidates {candidates} found in DataFrame columns.")
    return None

def safe_div(num: pd.Series | None, den: pd.Series | None) -> pd.Series:
    """
    Safely divide two Series, returning NaN if either is None.

    Args:
        num (pd.Series | None): Numerator Series.
        den (pd.Series | None): Denominator Series.
    Returns:
        pd.Series: Resulting Series after division or NaN if inputs are invalid.
    """
    if num is None or den is None:
        print("Warning: Cannot perform division, one of the Series is None.")
        return np.nan
    return num / den

def fetch_quarterly_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly fundamental data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.
    Returns:
        pd.DataFrame: DataFrame containing quarterly fundamental ratios with datetime index.
    Raises:
        ValueError: If no fundamental data is found for the given ticker.
    """
    tk = yf.Ticker(ticker)

    # raw statements (Yahoo only gives last 4-5 quarters)
    inc = tk.quarterly_financials
    bal = tk.quarterly_balance_sheet
    cf  = tk.quarterly_cashflow

    if inc.empty or bal.empty:
        raise ValueError(f"No fundamentals for {ticker}")
    
    # We traspose to have dates as index
    inc = inc.T
    bal = bal.T
    cf  = cf.T  

    # We ensure datetime index
    inc.index = pd.to_datetime(inc.index)
    bal.index = pd.to_datetime(bal.index)
    cf.index  = pd.to_datetime(cf.index)

    df = pd.DataFrame(index=inc.index)

    # Valuation ratios will be computed outside, as we need price data

    # Profitability ratios
    revenue = safe_col(inc, ["Total Revenue", "Revenue"])
    net_income = safe_col(inc, ["Net Income", "Net Income Common Stockholders"])
    operating_income = safe_col(inc, ["Operating Income", "Operating Income or Loss", "EBIT"])
    total_assets = safe_col(bal, ["Total Assets"])
    total_equity = safe_col(bal, ["Total Stockholder Equity", "Total Equity Gross Minority Interest"])
    ebitda = safe_col(inc, ["EBITDA"])

    df["net_margin"] = safe_div(net_income, revenue)
    df["operating_margin"] = safe_div(operating_income, revenue)
    df["ebitda_margin"] = safe_div(ebitda, revenue)
    df["asset_turnover"] = safe_div(revenue, total_assets)
    df["roe"] = safe_div(net_income, total_equity) # important to clip extreme values outside
    df["roa"] = safe_div(net_income, total_assets)

    # Growth ratios (yoy)
    fcf = safe_col(cf, ["Free Cash Flow", "Free Cash Flow Equity"])
    
    df["fcf_margin"] = safe_div(fcf, revenue)
    df["revenue_growth_qoq"] = (revenue.pct_change(1, fill_method=None) if revenue is not None else np.nan)
    df["earnings_growth_qoq"] = (net_income.pct_change(1, fill_method=None) if net_income is not None else np.nan)
    df["fcf_growth_qoq"] = (fcf.pct_change(1, fill_method=None) if fcf is not None else np.nan)

    # Risk ratios
    total_debt = safe_col(bal, ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"])
    current_assets = safe_col(bal, ["Current Assets"])
    current_liabilities = safe_col(bal, ["Current Liabilities"])
    #interest_expense = safe_col(inc, ["Interest Expense", "Interest Expense Non Operating"])

    df["debt_to_equity"] = safe_div(total_debt, total_equity)
    df["current_ratio"] = safe_div(current_assets, current_liabilities)
    #df["interest_coverage"] = safe_div(operating_income, interest_expense)

    # Clean to ensure proper types
    df = df.sort_index()

    #print(df) for debugging

    # Shift to ensure lagging, but before let's add a new row for the next quarter from the last available
    # next_quarter = df.index[-1] + pd.offsets.QuarterEnd(1)
    # df.loc[next_quarter] = df.iloc[-1]
    # df = df.shift(1)

    # As reports are normally released 45 to 60 days after quarter end, we shift by 60 days
    df.index = df.index + pd.Timedelta(days=60)
    
    #print(df) for debugging

    return df


# May add more functions for annual fundamentals, etc.