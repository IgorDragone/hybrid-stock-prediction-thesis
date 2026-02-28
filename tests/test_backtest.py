import numpy as np
import pandas as pd

from src.modeling.backtest import BacktestConfig, backtest_from_scores


def _base_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29", "2020-02-29"]
            ),
            "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "score": [0.9, 0.1, 0.2, 0.4, 0.8, 0.3],
            "fwd_ret_1m": [0.10, 0.02, -0.01, 0.05, -0.02, 0.01],
            "stress_index": [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        }
    )


def test_backtest_top_k_returns():
    df = _base_df()
    cfg = BacktestConfig(top_k=1, overlay_enabled=False)
    summary, artifacts = backtest_from_scores(df, {"model": "score"}, cfg)
    port_ret = artifacts["model"]["port_ret"]
    expected = pd.Series(
        [0.10, -0.02],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="port_ret",
    )
    assert np.allclose(port_ret.values, expected.values)
    assert summary.loc[summary["model"] == "model", "mean_monthly_return"].iloc[0] == np.mean(expected.values)


def test_backtest_overlay_scales_exposure():
    df = _base_df()
    cfg = BacktestConfig(top_k=1, overlay_enabled=True, stress_threshold=0.5, risk_off_exposure=0.6, cash_return=0.0)
    _, artifacts = backtest_from_scores(df, {"model": "score"}, cfg)
    port_ret = artifacts["model"]["port_ret"]
    # date1 has stress 0.8 -> exposure 0.6, date2 has stress 0.2 -> exposure 1.0
    expected = pd.Series(
        [0.10 * 0.6, -0.02 * 1.0],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="port_ret",
    )
    assert np.allclose(port_ret.values, expected.values)
