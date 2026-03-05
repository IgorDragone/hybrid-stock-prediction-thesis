import altair as alt
import pandas as pd
import streamlit as st

from src.app.logic.models import (
    benchmark_equity_curve,
    compare_on_subset,
    list_model_entries,
    model_equity_curve,
)

from .ui_components import status_box


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
                st.dataframe(summary, width="stretch", hide_index=True)
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
            st.dataframe(rows, width="stretch", hide_index=True)

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
