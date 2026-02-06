import streamlit as st

from ui_components import status_box


def compare_section():
    st.subheader("Model Comparison (Backtest Preview)")
    status_box(
        "Compare multiple models before choosing one. Metrics and charts are placeholders."
    )
    model_options = [
        "baseline_xgb_v1",
        "lstm_sequence_v2",
        "transformer_momentum_v1",
    ]
    selected = st.multiselect("Models to compare", model_options)
    if selected:
        st.dataframe(
            {
                "model": selected,
                "CAGR": ["12.4%", "10.1%", "13.0%"][: len(selected)],
                "Sharpe": ["1.05", "0.92", "1.10"][: len(selected)],
                "Max DD": ["-18%", "-21%", "-16%"][: len(selected)],
            }
        )
        st.line_chart(
            {
                "Baseline": [1.0, 1.02, 1.04, 1.03, 1.06],
                "Alt": [1.0, 1.01, 1.03, 1.02, 1.05],
            }
        )
