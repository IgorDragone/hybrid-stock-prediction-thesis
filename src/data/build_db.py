# src/data/build_db.py
import os
import pandas as pd

from config import START_DATE, END_DATE, TICKERS, FRED_SERIES, DATA_DIR
from prices import fetch_prices
from technicals import technical_indicators
from fundamentals import fetch_quarterly_fundamentals
from macro import fetch_macro_fred

def merge_macro_with_prices(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame: 
    """
    Merge macroeconomic data with price data using forward fill to align dates.

    Args:
        prices (pd.DataFrame): DataFrame containing price data with a DateTime index.
        macro (pd.DataFrame): DataFrame containing macroeconomic data with a DateTime index.
    Returns:
        pd.DataFrame: Merged DataFrame with macroeconomic data aligned to price dates.
    """
    macro = macro.copy()
    macro = macro.reindex(prices.index, method="ffill")
    #macro = macro.sort_index().ffill()

    SHIFT_RULES = {
        'GDP': 60,
        'CPIAUCSL': 15,
        'FEDFUNDS': 0,
    }

    for col, shift in SHIFT_RULES.items():
        if col in macro.columns and shift > 0:
            macro[col] = macro[col].shift(shift, freq='D')

    macro = macro.sort_index().ffill()
    df = prices.join(macro, how="left")

    return df


def build_database(tickers: list, start: str, end: str, out_path: str) -> None:
    """
    Build a database of financial data for given tickers and macroeconomic indicators.

    Args:
        tickers (list): List of stock ticker symbols.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        out_path (str): Path to save the resulting database parquet file.
    Returns:
        None
    """
    macro = fetch_macro_fred(
        FRED_SERIES,
        start, end
    )

    all_frames = []

    for ticker in tickers:
        print(f"Processing {ticker}")

        prices = fetch_prices(ticker, start, end)
        tech = technical_indicators(prices)

        df = prices.join(tech)

        # Merge fundamentals
        fundamentals = fetch_quarterly_fundamentals(ticker)
        fundamentals_daily = fundamentals.reindex(df.index, method="ffill") 
        df = df.join(fundamentals_daily)

        # Merge macro
        df = merge_macro_with_prices(df, macro)

        df["ticker"] = ticker
        all_frames.append(df)

    panel = pd.concat(all_frames)
    panel = panel.reset_index().rename(columns={'index': 'date'})
    panel = panel.sort_values(["date", "ticker"])

    panel.to_parquet(out_path, index=False)
    print(f"Saved database to {out_path}")

# main
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, "financial_database.parquet")
    build_database(TICKERS, START_DATE, END_DATE, out_file)
