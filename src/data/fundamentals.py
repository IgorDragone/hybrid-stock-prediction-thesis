# src/data/fundamentals.py
import yfinance as yf
import pandas as pd
import numpy as np

def safe_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None

def safe_div(num, den):
    if num is None or den is None:
        return np.nan
    return num / den

def fetch_quarterly_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Quarterly fundamentals snapshot (lagged).
    Output index: quarter end date
    Columns: valuation, profitability, growth, risk metrics
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

    df["net_margin"] = safe_div(net_income, revenue)
    df["operating_margin"] = safe_div(operating_income, revenue)
    df["roe"] = safe_div(net_income, total_equity)
    df["roa"] = safe_div(net_income, total_assets)

    # Growth ratios (quarter over quarter)
    fcf = safe_col(cf, ["Free Cash Flow", "Free Cash Flow Equity"])
    
    df["revenue_growth_yoy"] = (revenue.pct_change(4, fill_method=None) if revenue is not None else np.nan)
    df["earnings_growth_yoy"] = (net_income.pct_change(4, fill_method=None) if net_income is not None else np.nan)
    df["fcf_growth_yoy"] = (fcf.pct_change(4, fill_method=None) if fcf is not None else np.nan)

    # Risk ratios
    total_debt = safe_col(bal, ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"])
    current_assets = safe_col(bal, ["Current Assets"])
    current_liabilities = safe_col(bal, ["Current Liabilities"])
    interest_expense = safe_col(inc, ["Interest Expense", "Interest Expense Non Operating"])

    df["debt_to_equity"] = safe_div(total_debt, total_equity)
    df["current_ratio"] = safe_div(current_assets, current_liabilities)
    df["interest_coverage"] = safe_div(operating_income, interest_expense)

    shares_outstanding = tk.info.get("sharesOutstanding", np.nan)
    df["shares_outstanding"] = shares_outstanding

    # Clean to ensure proper types
    df = df.sort_index()


    # Shift to ensure lagging, but before let's add a new row for the next quarter from the last available
    next_quarter = df.index[-1] + pd.offsets.QuarterEnd(1)
    df.loc[next_quarter] = df.iloc[-1]
    df = df.shift(1)


    return df


# May add more functions for annual fundamentals, etc.