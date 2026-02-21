# src/db_building/config.py
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

# Dates and tickers are defined by the user in the UI or in run-spec YAML configs.

# -------------------------
# Macro (FRED)
# -------------------------
FRED_SERIES = [
    "CPIAUCSL",  # monthly
    "INDPRO",    # monthly
    "DGS10",     # daily
    "DGS2",      # daily
    "STLFSI4",   # weekly
]

# Data availability lag (days) applied uniformly after monthly alignment.
MACRO_LAG_DAYS = 30


# -------------------------
# Fundamentals (AlphaVantage)
# -------------------------
AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FUNDAMENTALS_LAG_DAYS = 60  # reporting delay approximation
