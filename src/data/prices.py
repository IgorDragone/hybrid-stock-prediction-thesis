# src/data/prices.py

import yfinance as yf
import pandas as pd

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical price data with columns:
                      ['open', 'high', 'low', 'close', 'adj_close', 'volume'] and index as 'date'.
    Raises:
        ValueError: If no price data is found for the given ticker.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    
    # Flatten MultiIndex (yf returns it even for 1 ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.index.name = "date"

    return df