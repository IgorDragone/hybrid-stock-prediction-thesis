# src/data/prices.py

import yfinance as yf
import pandas as pd

def fetch_prices(ticker, start, end):
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