import streamlit as st
from . import compare_flow, db_flow, model_flow, portfolio_flow, summary_page
from .ui_components import apply_style, status_box


def _init_flow_state():
    if "ui_page" not in st.session_state:
        st.session_state.ui_page = "portfolio_home"

    if "portfolio_name" not in st.session_state:
        st.session_state.portfolio_name = "My Portfolio"
    if "portfolio_tickers" not in st.session_state:
        st.session_state.portfolio_tickers = []
    if "rebalance_freq" not in st.session_state:
        st.session_state.rebalance_freq = "Monthly"

    if "model_mode" not in st.session_state:
        st.session_state.model_mode = "Load existing model"
    if "model_selected" not in st.session_state:
        st.session_state.model_selected = None
    if "model_candidate" not in st.session_state:
        st.session_state.model_candidate = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "available_models" not in st.session_state:
        st.session_state.available_models = [
            "baseline_xgb_v1",
            "lstm_sequence_v2",
            "transformer_momentum_v1",
        ]
    if "model_db_map" not in st.session_state:
        st.session_state.model_db_map = {}
    if "model_pending" not in st.session_state:
        st.session_state.model_pending = None

    if "db_mode" not in st.session_state:
        st.session_state.db_mode = "Load existing database"
    if "db_selected" not in st.session_state:
        st.session_state.db_selected = None
    if "db_candidate" not in st.session_state:
        st.session_state.db_candidate = None
    if "db_params" not in st.session_state:
        st.session_state.db_params = {"horizon": "3 Months", "tickers": 10}
    if "db_confirmed" not in st.session_state:
        st.session_state.db_confirmed = False

    if "saved_portfolios" not in st.session_state:
        st.session_state.saved_portfolios = []


def _summary_box(show_model: bool = False):
    model_label = st.session_state.model_selected or "Yet to choose"
    tickers = (
        ", ".join(st.session_state.portfolio_tickers)
        if st.session_state.portfolio_tickers
        else "None"
    )
    lines = [
        "<b>Summary</b>",
        f"Portfolio: {st.session_state.portfolio_name}",
        f"Tickers: {tickers}",
        f"Rebalance: {st.session_state.rebalance_freq}",
    ]
    if show_model:
        lines.append(f"Model: {model_label}")
        if st.session_state.model_selected:
            db_label = st.session_state.db_selected or st.session_state.model_db_map.get(
                st.session_state.model_selected, "Pending"
            )
            lines.append(f"Database used for training the model: {db_label}")
    status_box("<br>".join(lines))


def _model_page():
    st.subheader("Model Selection")
    _summary_box()

    tab_select, tab_compare = st.tabs(["Select Model", "Compare Models"])
    with tab_select:
        st.session_state.model_mode = st.radio(
            "Choose model path",
            ["Load existing model", "Create new model"],
            index=0 if st.session_state.model_mode == "Load existing model" else 1,
        )
        model_flow.render(st.session_state.model_mode)
    with tab_compare:
        compare_flow.compare_section()

    if st.button("Back to Portfolio", use_container_width=True):
        st.session_state.ui_page = "portfolio_builder"


def _db_page():
    st.subheader("Database Selection")
    _summary_box()
    if st.session_state.model_mode != "Create new model":
        status_box("Database selection is only available when creating a new model.")
        if st.button("Back to Models", use_container_width=True):
            st.session_state.ui_page = "model_page"
        return

    db_flow.render(st.session_state.db_mode)

    if st.session_state.db_confirmed:
        st.session_state.db_confirmed = False
        if st.session_state.model_pending:
            base_name = st.session_state.model_pending
            suffix = len(st.session_state.available_models) + 1
            new_name = f"{base_name}_custom_{suffix}"
            st.session_state.available_models.append(new_name)
            st.session_state.model_db_map[new_name] = st.session_state.db_selected
            st.session_state.model_pending = None
            st.session_state.model_mode = "Load existing model"
            st.session_state.model_candidate = new_name
            st.session_state.db_selected = None
        st.session_state.ui_page = "model_page"
        st.rerun()

    if st.button("Back to Models", use_container_width=True):
        st.session_state.ui_page = "model_page"




def render():
    _init_flow_state()
    apply_style()

    if st.session_state.ui_page == "portfolio_home":
        portfolio_flow.portfolio_home()
    elif st.session_state.ui_page == "portfolio_builder":
        portfolio_flow.portfolio_builder(_summary_box)
    elif st.session_state.ui_page == "model_page":
        _model_page()
    elif st.session_state.ui_page == "db_page":
        _db_page()
    elif st.session_state.ui_page == "summary_page":
        summary_page.summary_page(_summary_box)
