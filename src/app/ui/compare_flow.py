import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app.logic.models import (
    benchmark_equity_curve,
    compare_on_subset,
    list_model_entries,
    model_equity_curve,
)

from .ui_components import status_box


def _render_metrics_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        return

    table = df.copy()
    if "model" in table.columns and "mean_turnover" in table.columns:
        # Buy & Hold turnover is not comparable with active strategies in this UI context.
        bh_mask = table["model"].astype(str).str.contains("Buy & Hold|buy_hold_eqw", case=False, regex=True)
        table.loc[bh_mask, "mean_turnover"] = np.nan

    rename_map = {
        "model": "Model",
        "mean_monthly_return": "Mean Monthly Return",
        "vol_monthly": "Monthly Volatility",
        "sharpe": "Sharpe Ratio",
        "cagr": "CAGR",
        "max_drawdown": "Max Drawdown",
        "mean_turnover": "Mean Turnover",
        "hit_rate": "Hit Rate",
    }
    table = table.rename(columns=rename_map)

    numeric_cols = [c for c in table.columns if c != "Model" and pd.api.types.is_numeric_dtype(table[c])]
    best_direction = {
        "Mean Monthly Return": "max",
        "Monthly Volatility": "min",
        "Sharpe Ratio": "max",
        "CAGR": "max",
        "Max Drawdown": "max",  # less negative is better
        "Mean Turnover": "min",
        "Hit Rate": "max",
    }

    def _rounded_for_display(series: pd.Series, col_name: str) -> pd.Series:
        if col_name in {"Mean Monthly Return", "Monthly Volatility", "CAGR", "Max Drawdown", "Mean Turnover", "Hit Rate"}:
            return series.round(3)  # matches {:+.1%} display precision
        if col_name == "Sharpe Ratio":
            return series.round(2)
        return series

    def _highlight_best(col: pd.Series) -> list[str]:
        if col.name not in best_direction or col.dropna().empty:
            return [""] * len(col)
        shown = _rounded_for_display(col, col.name)
        target = shown.max() if best_direction[col.name] == "max" else shown.min()
        return [
            "color: #43a047; font-weight: 700" if pd.notna(v) and v == target else ""
            for v in shown
        ]

    fmt = {}
    for col in numeric_cols:
        if col in {"Mean Monthly Return", "Monthly Volatility", "CAGR", "Max Drawdown", "Mean Turnover", "Hit Rate"}:
            fmt[col] = "{:+.1%}"
        elif col == "Sharpe Ratio":
            fmt[col] = "{:+.2f}"

    styled = table.style.format(fmt, na_rep="—")
    for col in numeric_cols:
        styled = styled.apply(_highlight_best, subset=[col])

    st.dataframe(styled, width="stretch", hide_index=True)


def compare_section():
    st.subheader("Model Comparison (Backtest Preview)")
    entries = list_model_entries()
    if not entries:
        status_box("No saved models found. Train or load models to compare.")
        return

    label_map = {
        "baseline_mom": "Momentum Baseline",
        "ridge": "Ridge (Linear)",
        "hgb": "Gradient Boosting (HGB)",
        "buy_hold_eqw": "Buy & Hold (Equal Weight)",
    }
    model_options = [m.get("id") for m in entries]
    selected = st.multiselect("Models to compare", model_options, default=model_options)
    benchmark_label = {
        "sp500": "S&P 500",
        "nasdaq": "NASDAQ (QQQ)",
    }
    benchmark_options = []
    for benchmark_id in ("sp500", "nasdaq"):
        if benchmark_equity_curve(benchmark_id):
            benchmark_options.append(benchmark_id)
    selected_benchmarks = st.multiselect(
        "Benchmarks",
        benchmark_options,
        default=benchmark_options,
        format_func=lambda bid: benchmark_label.get(bid, bid),
    )
    if selected:
        tickers = st.session_state.get("watchlist_tickers", [])
        if tickers:
            st.caption("Metrics computed on the selected ticker subset.")
            if st.button("Run subset backtest", width="stretch"):
                with st.spinner("Computing subset backtest..."):
                    try:
                        summary, curves_df = _cached_compare_on_subset(tuple(selected), tuple(tickers))
                    except Exception as exc:  # noqa: BLE001
                        status_box(f"Unable to compute subset metrics: {exc}")
                        return
            else:
                return
            if not summary.empty:
                if "model" in summary.columns:
                    cols = ["model"] + [c for c in summary.columns if c != "model"]
                    summary = summary[cols]
                if "model" in summary.columns:
                    summary["model"] = summary["model"].map(lambda m: label_map.get(m, m))
                _render_metrics_table(summary)
            else:
                status_box("No metrics available for the selected models.")
        else:
            rows = []
            for entry in entries:
                if entry.get("id") not in selected:
                    continue
                metrics = entry.get("metrics", {})
                rows.append(
                    {
                        "model": entry.get("id"),
                        "cagr": metrics.get("cagr"),
                        "sharpe": metrics.get("sharpe"),
                        "max_drawdown": metrics.get("max_drawdown"),
                        "hit_rate": metrics.get("hit_rate"),
                    }
                )
            if rows:
                for row in rows:
                    row["model"] = label_map.get(row["model"], row["model"])
            _render_metrics_table(pd.DataFrame(rows))

        curves = {}
        if tickers:
            curves = {k: v for k, v in curves_df.items()}
        else:
            for model_id in selected:
                curve = model_equity_curve(model_id)
                if curve:
                    curves[model_id] = curve
        for benchmark_id in selected_benchmarks:
            curve = benchmark_equity_curve(benchmark_id)
            if curve:
                curves[benchmark_label.get(benchmark_id, benchmark_id)] = curve
        if curves:
            st.markdown("### Equity Curves")
            series = {}
            for model_id, curve in curves.items():
                if isinstance(curve, dict):
                    series[model_id] = pd.Series(
                        curve["equity"],
                        index=pd.to_datetime(curve["date"]),
                        name=model_id,
                    )
                else:
                    series[model_id] = pd.Series(
                        curve["equity"].values,
                        index=pd.to_datetime(curve.index),
                        name=model_id,
                    )
            df_curves = pd.concat(series.values(), axis=1).sort_index()
            df_curves = df_curves.dropna(axis=0, how="any")
            df_curves = df_curves / df_curves.iloc[0]
            df_long = (
                df_curves.reset_index()
                .rename(columns={"index": "date"})
                .melt(id_vars="date", var_name="model", value_name="equity")
            )
            df_long["equity"] = df_long["equity"] * 1000
            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(format="%Y", tickCount=8, title="Year")),
                    y=alt.Y("equity:Q", axis=alt.Axis(title="Equity ($)")),
                    color=alt.Color("model:N", legend=alt.Legend(title="Model")),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
            st.caption("X: year · Y: equity value (starting from $1000)")
        else:
            status_box("No equity curves found for selected models.")


@st.cache_data(show_spinner=False)
def _cached_compare_on_subset(model_ids: tuple[str, ...], tickers: tuple[str, ...]):
    return compare_on_subset(list(model_ids), list(tickers))
