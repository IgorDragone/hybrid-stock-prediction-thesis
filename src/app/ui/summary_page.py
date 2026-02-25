import streamlit as st
from .ui_components import header, status_box
from src.app.logic.portfolio import save_portfolios
from src.app.logic.recommendations import score_portfolio

MAX_PORTFOLIOS = 3


def summary_page(summary_fn):
    header(
        "Portfolio Summary",
        "Review the portfolio configuration and decisions before saving.",
    )
    summary_fn(show_model=True)

    st.subheader("Decisions (Preview)")
    try:
        result = score_portfolio(
            model_id=st.session_state.model_selected,
            tickers=st.session_state.portfolio_tickers,
            cash=st.session_state.portfolio_cash,
        )
        st.markdown(f"**As of:** {result['date'].date()}")
        st.markdown(f"**Exposure:** {result['exposure']:.0%}")
        if result.get("stress_index") is not None:
            st.markdown(f"**Stress index:** {result['stress_index']:.2f}")
        st.markdown(f"**Cash left:** {result['cash_left']:,.0f}")
        st.dataframe(result["recommendations"], use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        status_box(f"Unable to compute recommendations: {exc}")

    st.subheader("Save Portfolio")
    if len(st.session_state.saved_portfolios) >= MAX_PORTFOLIOS:
        names = [p["name"] for p in st.session_state.saved_portfolios]
        if st.session_state.portfolio_name not in names:
            status_box("Maximum portfolios reached. Delete one to save a new portfolio.")
            st.button("Save Portfolio", use_container_width=True, disabled=True)
            return

    if st.button("Save Portfolio", use_container_width=True):
        existing = None
        for p in st.session_state.saved_portfolios:
            if p["name"] == st.session_state.portfolio_name:
                existing = p
                break
        payload = {
            "name": st.session_state.portfolio_name,
            "tickers": st.session_state.portfolio_tickers,
            "rebalance": st.session_state.rebalance_freq,
            "cash": st.session_state.portfolio_cash,
            "model": st.session_state.model_selected,
            "database": st.session_state.db_selected,
        }
        if existing:
            existing.update(payload)
        else:
            st.session_state.saved_portfolios.append(payload)
        save_portfolios(st.session_state.saved_portfolios)
        st.success("Portfolio saved.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Models", use_container_width=True):
            st.session_state.ui_page = "model_page"
    with col2:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.ui_page = "portfolio_home"
