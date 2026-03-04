import pandas as pd

from src.db_building.build_db import merge_fundamentals_asof
from src.db_building.technicals import technical_indicators


def test_fundamentals_effective_date_no_leakage():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-20"]),
            "adj_close": [100.0, 101.0, 102.0],
        }
    )
    fundamentals_q = pd.DataFrame(
        {
            "fiscalDateEnding": pd.to_datetime(["2019-12-31"]),
            "roe": [0.1],
        }
    )
    merged = merge_fundamentals_asof(daily, fundamentals_q, lag_days=5)
    # If a row has fundamentals attached, its effective_date should not be after the daily date.
    valid = merged.dropna(subset=["roe"])
    assert (valid["effective_date"] <= valid["date"]).all()


def test_technicals_shifted_returns():
    df = pd.DataFrame({"adj_close": [100, 102, 101, 105, 110]})
    out = technical_indicators(df)
    expected = df["adj_close"].pct_change(1).shift(1)
    diff = (out["ret_1d"] - expected).abs().dropna()
    assert diff.max() < 1e-12
