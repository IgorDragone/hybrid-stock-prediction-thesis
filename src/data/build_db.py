# src/data/build_db.py
import os
import pandas as pd

from config import START_DATE, END_DATE, TICKERS, FRED_SERIES, DATA_DIR
from prices import fetch_prices
from technicals import technical_indicators
from fundamentals import fetch_quarterly_fundamentals
from macro import fetch_macro_fred

def build_database(tickers: list, start: str, end: str, out_path: str) -> None:
    macro = fetch_macro_fred(
        FRED_SERIES,
        start, end
    )# .reset_index().rename(columns={'index': 'date'})

    all_frames = []

    for ticker in tickers:
        print(f"Processing {ticker}")

        prices = fetch_prices(ticker, start, end)
        tech = technical_indicators(prices)

        df = prices.join(tech)

        fundamentals = fetch_quarterly_fundamentals(ticker)
        fundamentals_daily = fundamentals.reindex(df.index, method="ffill")
        df = df.join(fundamentals_daily)

        macro_daily = macro.reindex(df.index, method="ffill")
        df = df.join(macro_daily)

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
