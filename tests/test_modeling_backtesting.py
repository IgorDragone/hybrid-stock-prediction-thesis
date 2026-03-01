import numpy as np
import pandas as pd

from src.modeling.backtest import BacktestConfig, backtest_from_scores
from src.modeling.splits import WalkForwardConfig, generate_expanding_walk_forward_splits


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
    expected = pd.Series(
        [0.10 * 0.6, -0.02 * 1.0],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="port_ret",
    )
    assert np.allclose(port_ret.values, expected.values)


def test_walk_forward_embargo_respected():
    dates = pd.date_range("2018-01-31", periods=36, freq="M")
    df = pd.DataFrame({"date": dates, "ticker": ["AAA"] * len(dates)})
    cfg = WalkForwardConfig(train_years=1, test_months=3, embargo_months=2, min_train_months=12)
    splits = list(generate_expanding_walk_forward_splits(df, cfg))
    assert splits, "Expected at least one split"
    for _, _, info in splits:
        train_end = pd.to_datetime(info["train_end"]).to_period("M")
        test_start = pd.to_datetime(info["test_start"]).to_period("M")
        assert test_start >= train_end + cfg.embargo_months
