import pandas as pd

from src.preprocessing.target_construction import TargetConfig, construct_target


def test_target_sanity_demeaning():
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "AAA", 100 + i * 2.0))
        rows.append((d, "BBB", 100 + i * 1.0))
        rows.append((d, "CCC", 100 - i * 0.5))

    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    cfg = TargetConfig(horizon_1m=1, horizon_3m=2)
    out = construct_target(df, cfg)

    target = out[cfg.target_3m_col].dropna()
    assert not target.empty

    cs_mean = out.groupby("date")[cfg.target_3m_col].mean().dropna()
    assert (cs_mean.abs() < 1e-12).all()
