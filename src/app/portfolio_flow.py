import streamlit as st
from ui_components import header, status_box

MAX_PORTFOLIOS = 3


def reset_current_portfolio():
    st.session_state.portfolio_name = "My Portfolio"
    st.session_state.portfolio_tickers = []
    st.session_state.rebalance_freq = "Monthly"
    st.session_state.model_selected = None
    st.session_state.db_selected = None


def load_portfolio(p):
    st.session_state.portfolio_name = p["name"]
    st.session_state.portfolio_tickers = p["tickers"]
    st.session_state.rebalance_freq = p["rebalance"]
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
                    f"{p['rebalance']} | model: {p['model']}"
                )
            with col2:
                if st.button("Open", key=f"open_{idx}", use_container_width=True):
                    load_portfolio(p)
                    st.session_state.ui_page = "summary_page"
            with col3:
                if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                    st.session_state.saved_portfolios.pop(idx)
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

    sector_map = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
        "Consumer": ["AMZN", "KO"],
        "Healthcare": ["JNJ"],
        "Energy": ["XOM"],
        "Auto": ["TSLA"],
    }
    sectors = list(sector_map.keys())
    selected_sector = st.selectbox(
        "Filter by sector",
        ["All"] + sectors,
        index=0,
    )
    if selected_sector == "All":
        available = sorted({t for v in sector_map.values() for t in v})
    else:
        available = sector_map[selected_sector]
    available = sorted(set(available + st.session_state.portfolio_tickers))

    tickers = st.multiselect(
        "Select tickers (max 8)",
        available,
        default=st.session_state.portfolio_tickers,
        key="portfolio_ticker_select",
    )
    if len(tickers) > 8:
        st.warning("Maximum 8 tickers allowed. Please remove extras.")
        tickers = tickers[:8]
    st.session_state.portfolio_tickers = tickers

    st.session_state.rebalance_freq = st.selectbox(
        "Rebalancing frequency",
        ["Monthly", "Quarterly"],
        index=0 if st.session_state.rebalance_freq == "Monthly" else 1,
    )
    summary_fn()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state.ui_page = "portfolio_home"
    invalid_count = len(st.session_state.portfolio_tickers) == 0 or len(st.session_state.portfolio_tickers) > 8
    with col2:
        if st.button(
            "Continue to Models",
            use_container_width=True,
            disabled=invalid_count,
        ):
            st.session_state.ui_page = "model_page"
