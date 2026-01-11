# src/data/config.py

from datetime import datetime

# Dates
START_DATE = "2015-01-01" 
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Tickers
TICKERS = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Non-tech
    "JPM", "JNJ", "XOM"
]

# Macro (FRED)
FRED_SERIES = [
    "GDP",        # quarterly
    "CPIAUCSL",   # monthly
    "FEDFUNDS",   # monthly
]

# Paths
DATA_DIR = "data/processed"