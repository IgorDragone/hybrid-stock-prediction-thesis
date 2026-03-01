import streamlit as st

from src.app.logic.portfolio import save_portfolios
from src.config import load_universe

from .ui_components import header, status_box

MAX_PORTFOLIOS = 3
MAX_TICKERS = 15


def reset_current_portfolio():
    st.session_state.portfolio_name = "My Portfolio"
    st.session_state.portfolio_tickers = []
    st.session_state.rebalance_freq = "Monthly"
    st.session_state.portfolio_cash = 1000.0
    st.session_state.model_selected = None
    st.session_state.db_selected = None


def load_portfolio(p):
    st.session_state.portfolio_name = p["name"]
    st.session_state.portfolio_tickers = p["tickers"]
    st.session_state.rebalance_freq = p.get("rebalance", "Monthly")
    st.session_state.portfolio_cash = p.get("cash", 1000.0)
    st.session_state.model_selected = p["model"]
    st.session_state.db_selected = p["database"]


def portfolio_home():
    header(
        "Portfolio Hub",
        "Open an existing portfolio or create a new one.",
    )
    if st.session_state.saved_portfolios:
        st.markdown("### Saved Portfolios")
        for idx, p in enumerate(st.session_state.saved_portfolios):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(
                    f"**{p['name']}** — {', '.join(p['tickers'])} | "
                    f"cash: {p.get('cash', 0):,.0f} | model: {p['model']}"
                )
            with col2:
                if st.button("Open", key=f"open_{idx}", use_container_width=True):
                    load_portfolio(p)
                    st.session_state.ui_page = "summary_page"
            with col3:
                if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                    st.session_state.saved_portfolios.pop(idx)
                    save_portfolios(st.session_state.saved_portfolios)
                    st.rerun()
    else:
        status_box("No saved portfolios yet. Create a new one to get started.")

    if len(st.session_state.saved_portfolios) >= MAX_PORTFOLIOS:
        status_box("Maximum portfolios reached. Delete one to create a new portfolio.")
        st.button("Create New Portfolio", use_container_width=True, disabled=True)
    else:
        if st.button("Create New Portfolio", use_container_width=True):
            reset_current_portfolio()
            st.session_state.ui_page = "portfolio_builder"


def portfolio_builder(summary_fn):
    header(
        "Portfolio Builder",
        "Select tickers and configuration before choosing a model.",
    )
    st.subheader("Portfolio Details")
    st.session_state.portfolio_name = st.text_input(
        "Portfolio name",
        value=st.session_state.portfolio_name,
    )
    st.session_state.portfolio_cash = st.number_input(
        "Initial cash (EUR)",
        min_value=100.0,
        max_value=1_000_000.0,
        step=100.0,
        value=float(st.session_state.portfolio_cash),
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
    available = sorted(set(available + st.session_state.portfolio_tickers))

    tickers = st.multiselect(
        f"Select tickers (recommended 8–{MAX_TICKERS})",
        available,
        default=st.session_state.portfolio_tickers,
        key="portfolio_ticker_select",
    )
    if len(tickers) > MAX_TICKERS:
        st.warning(f"Maximum {MAX_TICKERS} tickers allowed. Please remove extras.")
    st.session_state.portfolio_tickers = tickers

    summary_fn()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state.ui_page = "portfolio_home"
    invalid_count = len(st.session_state.portfolio_tickers) == 0 or len(st.session_state.portfolio_tickers) > MAX_TICKERS
    with col2:
        if st.button(
            "Continue to Models",
            use_container_width=True,
            disabled=invalid_count,
        ):
            st.session_state.ui_page = "model_page"
