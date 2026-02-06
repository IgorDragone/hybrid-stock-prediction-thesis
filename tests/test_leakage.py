import pandas as pd

from src.db_buiding.technicals import technical_indicators


def test_fundamentals_effective_date_no_leakage():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-05"]),
            "effective_date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]),
        }
    )
    assert (df["effective_date"] <= df["date"]).all()


def test_technicals_shifted_returns():
    df = pd.DataFrame(
        {
            "adj_close": [100, 102, 101, 105, 110],
        }
    )
    out = technical_indicators(df)
    expected = df["adj_close"].pct_change(1).shift(1)
    diff = (out["ret_1d"] - expected).abs().dropna()
    assert diff.max() < 1e-12
