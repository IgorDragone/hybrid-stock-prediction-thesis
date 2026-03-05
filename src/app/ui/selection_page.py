import streamlit as st

from src.app.logic.watchlists import load_watchlists
from src.config import load_universe
from src.modeling.registry import list_models

from . import compare_flow, model_flow, summary_page, watchlist_flow
from .ui_components import apply_style, status_box


def _init_flow_state():
    if "ui_page" not in st.session_state:
        st.session_state.ui_page = "watchlist_home"

    if "watchlist_name" not in st.session_state:
        st.session_state.watchlist_name = "My Watchlist"
    if "watchlist_id" not in st.session_state:
        st.session_state.watchlist_id = None
    if "watchlist_tickers" not in st.session_state:
        st.session_state.watchlist_tickers = []

    if "model_selected" not in st.session_state:
        st.session_state.model_selected = None
    if "model_candidate" not in st.session_state:
        st.session_state.model_candidate = None
    if "available_models" not in st.session_state:
        st.session_state.available_models = list_models()
    if "db_selected" not in st.session_state:
        st.session_state.db_selected = None

    if "saved_watchlists" not in st.session_state:
        st.session_state.saved_watchlists = load_watchlists()


def _summary_box(show_model: bool = False):
    label_map = {
        "baseline_mom": "Momentum Baseline",
        "ridge": "Ridge (Linear)",
        "hgb": "Gradient Boosting (HGB)",
        "buy_hold_eqw": "Buy & Hold (Equal Weight)",
    }
    model_label = label_map.get(st.session_state.model_selected, st.session_state.model_selected) or "Yet to choose"
    tickers = (
        ", ".join(st.session_state.watchlist_tickers)
        if st.session_state.watchlist_tickers
        else "None"
    )
    tickers_by_sector = []
    if st.session_state.watchlist_tickers:
        universe = load_universe()
        sector_map = {entry.get("sector", "Unknown"): entry.get("tickers", []) for entry in universe}
        for sector, sector_tickers in sorted(sector_map.items()):
            selected = [t for t in sector_tickers if t in st.session_state.watchlist_tickers]
            if selected:
                tickers_by_sector.append(f"&nbsp;&nbsp;- {sector} ({len(selected)}): {', '.join(selected)}")
    lines = [
        "<b>Summary</b>",
        f"Name: {st.session_state.watchlist_name}",
        "Watchlist:",
    ]
    if tickers_by_sector:
        lines[3:3] = tickers_by_sector
    else:
        lines.insert(3, tickers)
    if show_model:
        lines.append(f"Model: {model_label}")
    status_box("<br>".join(lines))


def _model_page():
    st.subheader("Model Selection")
    _summary_box()

    tab_select, tab_compare = st.tabs(["Select Model", "Compare Models"])
    with tab_select:
        st.markdown("Pick one of the available models.")
        model_flow.render()
    with tab_compare:
        compare_flow.compare_section()

    if st.button("Back to Setup", width="stretch"):
        st.session_state.ui_page = "watchlist_builder"

def render():
    _init_flow_state()
    apply_style()

    if st.session_state.ui_page == "watchlist_home":
        watchlist_flow.watchlist_home()
    elif st.session_state.ui_page == "watchlist_builder":
        watchlist_flow.watchlist_builder(_summary_box)
    elif st.session_state.ui_page == "model_page":
        _model_page()
    elif st.session_state.ui_page == "summary_page":
        summary_page.summary_page(_summary_box)
    else:
        # Defensive fallback: avoid blank page if session state gets an unknown route.
        st.session_state.ui_page = "watchlist_home"
        status_box("Recovered invalid page state. Redirecting to Recommendation Hub.")
        st.rerun()
