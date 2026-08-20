from uuid import uuid4

import streamlit as st

from src.app.logic.watchlists import save_watchlists
from src.config import load_universe

from .ui_components import header, status_box

MAX_PORTFOLIOS = 3
MAX_TICKERS = 15


def reset_current_watchlist():
    st.session_state.watchlist_id = None
    st.session_state.watchlist_name = "My Watchlist"
    st.session_state.watchlist_tickers = []
    st.session_state.model_selected = None
    st.session_state.db_selected = None


def load_watchlist(p):
    if "id" not in p:
        p["id"] = str(uuid4())
    st.session_state.watchlist_id = p["id"]
    st.session_state.watchlist_name = p["name"]
    st.session_state.watchlist_tickers = p["tickers"]
    st.session_state.model_selected = p["model"]
    st.session_state.db_selected = p["database"]


def watchlist_home():
    header(
        "Recommendation Hub",
        "Open a saved scenario or create a new recommendation setup.",
    )
    if st.session_state.saved_watchlists:
        updated_ids = False
        for p in st.session_state.saved_watchlists:
            if "id" not in p:
                p["id"] = str(uuid4())
                updated_ids = True
        if updated_ids:
            save_watchlists(st.session_state.saved_watchlists)
        st.markdown("### Watchlists")
        for idx, p in enumerate(st.session_state.saved_watchlists):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(
                    f"**{p['name']}** — {', '.join(p['tickers'])} | "
                    f"model: {p['model']}"
                )
            with col2:
                if st.button("Open", key=f"open_{idx}", width="stretch"):
                    load_watchlist(p)
                    st.session_state.ui_page = "summary_page"
            with col3:
                if st.button("Edit setup", key=f"edit_{idx}", width="stretch"):
                    load_watchlist(p)
                    st.session_state.ui_page = "watchlist_builder"
            with col4:
                if st.button("Delete", key=f"delete_{idx}", width="stretch"):
                    st.session_state.saved_watchlists.pop(idx)
                    save_watchlists(st.session_state.saved_watchlists)
                    st.rerun()
    else:
        status_box("No saved watchlists yet. Create one to get started.")

    if len(st.session_state.saved_watchlists) >= MAX_PORTFOLIOS:
        status_box("Maximum watchlists reached. Delete one to create a new setup.")
        st.button("Create New Watchlist", width="stretch", disabled=True)
    else:
        if st.button("Create New Watchlist", width="stretch"):
            reset_current_watchlist()
            st.session_state.ui_page = "watchlist_builder"


def watchlist_builder(summary_fn):
    header(
        "Recommendation Setup",
        "Select a watchlist before choosing a model.",
    )
    st.subheader("Watchlist Details")
    st.session_state.watchlist_name = st.text_input(
        "Watchlist name",
        value=st.session_state.watchlist_name,
    )
    universe = load_universe()
    sector_map = {entry.get("sector", "Unknown"): entry.get("tickers", []) for entry in universe}
    sectors = sorted(sector_map.keys())
    selected_sectors = st.multiselect(
        "Filter by sector (optional)",
        sectors,
        default=[],
    )
    if selected_sectors:
        available = sorted({t for s in selected_sectors for t in sector_map.get(s, [])})
    else:
        available = sorted({t for v in sector_map.values() for t in v})
    available = sorted(set(available + st.session_state.watchlist_tickers))

    tickers = st.multiselect(
        f"Select tickers (recommended 8–{MAX_TICKERS})",
        available,
        default=st.session_state.watchlist_tickers,
        key="watchlist_ticker_select",
    )
    if len(tickers) > MAX_TICKERS:
        st.warning(f"Maximum {MAX_TICKERS} tickers allowed. Please remove extras.")
    st.session_state.watchlist_tickers = tickers

    summary_fn()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", width="stretch"):
            st.session_state.ui_page = "watchlist_home"
    invalid_count = len(st.session_state.watchlist_tickers) == 0 or len(st.session_state.watchlist_tickers) > MAX_TICKERS
    with col2:
        if st.button(
            "Continue to Models",
            width="stretch",
            disabled=invalid_count,
        ):
            st.session_state.ui_page = "model_page"
