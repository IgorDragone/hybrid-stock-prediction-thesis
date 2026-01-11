# src/data/technicals.py

import pandas as pd
import numpy as np

def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    price = df["adj_close"]

    out = pd.DataFrame(index=df.index)

    # Trend
    out["sma_50"] = price.rolling(50).mean()
    out["sma_200"] = price.rolling(200).mean()
    out["ema_20"] = price.ewm(span=20, adjust=False).mean()

    out["price_sma_50"] = price / out["sma_50"]

    # Momentum
    delta = price.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Volatility
    out["volatility_20d"] = price.pct_change().rolling(20).std()

#     high_low = df["high"] - df["low"]
#     high_close = np.abs(df["high"] - df["close"].shift())
#     low_close = np.abs(df["low"] - df["close"].shift())

#     tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#     df["atr_14"] = tr.rolling(14).mean()

#     rolling_mean = price.rolling(20).mean()
#     rolling_std = price.rolling(20).std()
#     upper = rolling_mean + 2 * rolling_std
#     lower = rolling_mean - 2 * rolling_std
#     df["bb_width"] = (upper - lower) / price

    # Historical returns
    out["ret_1d"] = price.pct_change(1)
    out["ret_5d"] = price.pct_change(5)
    out["ret_21d"] = price.pct_change(21)

    # Shift
    out = out.shift(1)

    return out
