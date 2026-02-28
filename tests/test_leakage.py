import pandas as pd

from src.db_building.technicals import technical_indicators


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
