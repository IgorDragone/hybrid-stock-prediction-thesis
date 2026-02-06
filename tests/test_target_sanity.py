import numpy as np
import pandas as pd

from src.preprocessing.target_construction import TargetConfig, construct_target


def test_target_sanity_distribution_and_monotonicity():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "AAA", 100 + i * 2.0))
        rows.append((d, "BBB", 100 + i * 1.0))
        rows.append((d, "CCC", 100 - i * 0.5))

    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    cfg = TargetConfig(horizon_3m=2, horizon_6m=3, min_points_cs=3)
    out = construct_target(df, cfg)

    target = out["target"].dropna()
    assert set(target.unique()).issubset({-1, 0, 1})

    counts = target.value_counts(normalize=True).to_dict()
    for k in [-1, 0, 1]:
        assert abs(counts.get(k, 0) - 1 / 3) < 0.2

    means = out.groupby("target")["fwd_ret_3_6m"].mean()
    assert means.loc[1] > means.loc[0] > means.loc[-1]
