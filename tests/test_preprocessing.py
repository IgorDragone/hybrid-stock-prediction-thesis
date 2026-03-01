import pandas as pd

from src.preprocessing.pipeline import _sample_eom
from src.preprocessing.features_engineering import engineer_features


def test_eom_sampling_one_row_per_ticker_month():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-15",
                    "2020-01-31",
                    "2020-02-10",
                    "2020-02-28",
                    "2020-01-20",
                    "2020-01-31",
                    "2020-02-05",
                    "2020-02-28",
                ]
            ),
            "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "adj_close": [1, 2, 3, 4, 10, 11, 12, 13],
        }
    )
    out = _sample_eom(df, date_col="date", ticker_col="ticker")
    counts = out.groupby([out["date"].dt.to_period("M"), "ticker"]).size()
    assert (counts == 1).all()
    # Last date per ticker/month should be kept
    expected_last = (
        df.groupby([df["date"].dt.to_period("M"), "ticker"])["date"].max().sort_values()
    )
    out_last = (
        out.groupby([out["date"].dt.to_period("M"), "ticker"])["date"].max().sort_values()
    )
    assert expected_last.equals(out_last)


def test_pr_features_fillna():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "DGS10": [1.5, 1.5, 1.6, 1.6],
            "DGS2": [0.5, 0.5, 0.6, 0.6],
            "STLFSI4": [0.1, 0.1, 0.1, 0.1],
            "ip_yoy": [1.0, 1.0, 1.1, 1.1],
            "cpi_yoy": [2.0, 2.0, 2.1, 2.1],
            "roe": [0.1, None, 0.2, None],
            "mom_12m": [0.2, None, 0.25, None],
            "volatility_60d": [0.05, None, 0.06, None],
        }
    )
    out = engineer_features(df, date_col="date", ticker_col="ticker")
    for col in ["roe_pr", "mom12_pr", "vol_pr"]:
        assert col in out.columns
        assert out[col].isna().sum() == 0
