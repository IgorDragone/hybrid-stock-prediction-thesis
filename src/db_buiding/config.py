# src/db_buiding/config.py
"""
Configuration for financial database building, including paths, API keys, and data lags.
"""
from __future__ import annotations

import os

from src.config.paths import DATA_DIR


# -------------------------
# Project paths (repo-relative)
# -------------------------
RAW_DIR = DATA_DIR / "raw"

RAW_PRICES_DIR = RAW_DIR / "prices"
RAW_MACRO_DIR = RAW_DIR / "macro"
RAW_AV_DIR = RAW_DIR / "alphavantage"  # json cache per endpoint/ticker

# Dates and tickers are defined in run-spec YAML configs.

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
