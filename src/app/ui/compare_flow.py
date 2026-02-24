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
        if curves:
            st.markdown("### Equity Curves")
            for model_id, curve in curves.items():
                st.line_chart(
                    {"equity": curve["equity"]},
                    use_container_width=True,
                )
        else:
            status_box("No equity curves found for selected models.")
