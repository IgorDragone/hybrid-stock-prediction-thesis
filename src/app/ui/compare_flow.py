import altair as alt
import pandas as pd
import streamlit as st
from .ui_components import status_box
from src.app.logic.models import list_model_entries, model_equity_curve


def compare_section():
    st.subheader("Model Comparison (Backtest Preview)")
    entries = list_model_entries()
    if not entries:
        status_box("No saved models found. Train or load models to compare.")
        return

    model_options = [m.get("id") for m in entries]
    selected = st.multiselect("Models to compare", model_options, default=model_options)
    if selected:
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
        st.dataframe(rows, use_container_width=True)

        curves = {}
        for model_id in selected:
            curve = model_equity_curve(model_id)
            if curve:
                curves[model_id] = curve
        missing_curves = [m for m in selected if m not in curves]
        if missing_curves:
            status_box(
                "Missing equity curves for: "
                + ", ".join(missing_curves)
                + ". Run notebook 03 to generate them."
            )
        if curves:
            st.markdown("### Equity Curves")
            # Build a single dataframe with aligned dates for all models.
            series = {}
            for model_id, curve in curves.items():
                series[model_id] = pd.Series(
                    curve["equity"],
                    index=pd.to_datetime(curve["date"]),
                    name=model_id,
                )
            df_curves = pd.concat(series.values(), axis=1).sort_index()
            df_long = (
                df_curves.reset_index()
                .rename(columns={"index": "date"})
                .melt(id_vars="date", var_name="model", value_name="equity")
            )
            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(format="%Y", tickCount=8, title="Year")),
                    y=alt.Y("equity:Q", axis=alt.Axis(title="Equity value")),
                    color=alt.Color("model:N", legend=alt.Legend(title="Model")),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("X: year · Y: equity value")
        else:
            status_box("No equity curves found for selected models.")
