import pandas as pd


def test_schema_stability_required_columns_and_types():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["AAA", "BBB"],
            "adj_close": [100.0, 101.0],
            "target": [1, -1],
        }
    )
    required_cols = ["date", "ticker", "adj_close", "target"]
    for c in required_cols:
        assert c in df.columns

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["ticker"].dtype == object
