# src/data/technicals.py
"""Module for calculating technical indicators from stock price data."""

import pandas as pd


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame of stock prices.

    Args:
        df (pd.DataFrame): DataFrame containing historical price data
        
    Returns:
        pd.DataFrame: DataFrame containing technical indicators with the same index as input df.
        The indicators include:
            - Trend Indicators: SMA (200), Price/SMA(200)
            - Momentum Indicators: 3m, 6m, 12m (skip last month) momentum
            - Volatility Indicators: Volatility (20d, 60d)
            - Historical Returns: 1d, 5d, 21d returns
    """

    if "adj_close" not in df.columns:
        raise ValueError("technicals.py: expected column 'adj_close' in input DataFrame.")

    price = df["adj_close"]

    out = pd.DataFrame(index=df.index)

    # Trend Indicators
    out["sma_200"] = price.rolling(200).mean()
    out["price_sma_200"] = price / out["sma_200"]

    # Momentum Indicators (approx trading days per month)
    tdm = 21
    out["mom_3m"] = price.shift(tdm) / price.shift(tdm + 3 * tdm) - 1
    out["mom_6m"] = price.shift(tdm) / price.shift(tdm + 6 * tdm) - 1
    out["mom_12m"] = price.shift(tdm) / price.shift(tdm + 12 * tdm) - 1

    # Volatility Indicators
    out["volatility_20d"] = price.pct_change().rolling(20).std()
    out["volatility_60d"] = price.pct_change().rolling(60).std()

    # Historical returns as features (for backtesting / diagnostics)
    out["ret_1d"] = price.pct_change(1)
    out["ret_5d"] = price.pct_change(5)
    out["ret_21d"] = price.pct_change(21)

    # Shift by 1 to avoid lookahead bias 
    out = out.shift(1)

    return out