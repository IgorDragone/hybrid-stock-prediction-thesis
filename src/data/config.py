# src/data/config.py
"""
Central configuration for data paths, dataset universe, and time ranges.

This module defines:
- Repository-relative paths for raw and processed data
- Data collection universe (tickers) and macro series
- Availability lags (days) to avoid look-ahead bias

Notes
-----
- Dates are expressed in ISO format: YYYY-MM-DD.
- Lags are approximate and project-specific; they are used to simulate data availability.
- API keys are read from environment variables to avoid hardcoding secrets.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime


# -------------------------
# Project paths (repo-relative)
# -------------------------
# This file is repo_root/src/data/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo_root/
DATA_ROOT = REPO_ROOT / "data"

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

RAW_PRICES_DIR = RAW_DIR / "prices"
RAW_MACRO_DIR = RAW_DIR / "macro"
RAW_AV_DIR = RAW_DIR / "alphavantage"  # json cache per endpoint/ticker

# -------------------------
# Dates
# -------------------------
START_DATE = "2015-01-01"

# We can set END_DATE to today or a fixed date for reproducibility
# END_DATE = datetime.today().strftime("%Y-%m-%d")
END_DATE = "2025-12-31" 

# -------------------------
# Universe (assets)
# -------------------------
TICKERS = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Non-tech
    "KO", "JNJ", "XOM",
]

# -------------------------
# Macro (FRED)
# -------------------------
FRED_SERIES = [
    "GDP",        # quarterly
    "CPIAUCSL",   # monthly
    "FEDFUNDS",   # monthly
]

# Data availability lags (days) used to prevent look-ahead bias.
MACRO_LAG_DAYS = {
    "GDP": 60,
    "CPIAUCSL": 15,
    "FEDFUNDS": 0, # assume immediate availability
}


# -------------------------
# Fundamentals (AlphaVantage)
# -------------------------
AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FUNDAMENTALS_LAG_DAYS = 60  # reporting delay approximation