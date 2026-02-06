import pandas as pd


def test_panel_integrity_no_duplicates_and_full_coverage():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    tickers = ["AAA", "BBB", "CCC"]
    rows = [(d, t) for d in dates for t in tickers]
    df = pd.DataFrame(rows, columns=["date", "ticker"])

    dupes = df.duplicated(subset=["date", "ticker"]).sum()
    assert dupes == 0

    counts = df.groupby("date")["ticker"].nunique()
    assert (counts == len(tickers)).all()
