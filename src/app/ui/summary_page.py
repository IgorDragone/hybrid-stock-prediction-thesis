import streamlit as st

from src.app.logic.portfolio import save_portfolios
from src.app.logic.recommendations import score_portfolio

from .ui_components import header, status_box

MAX_PORTFOLIOS = 3


def summary_page(summary_fn):
    header(
        "Portfolio Summary",
        "Review the portfolio configuration and decisions before saving.",
    )
    summary_fn(show_model=True)

    st.subheader("Decisions")
    try:
        if not st.session_state.model_selected:
            status_box("Select a model to generate portfolio decisions.")
            return
        result = score_portfolio(
            model_id=st.session_state.model_selected,
            tickers=st.session_state.portfolio_tickers,
            cash=st.session_state.portfolio_cash,
        )
        if result.get("used_fallback"):
            status_box("Model could not be loaded (joblib missing). Using baseline score instead.")
        st.markdown(f"**Exposure:** {result['exposure']:.0%}")
        if result.get("stress_index") is not None:
            st.markdown(f"**Stress index:** {result['stress_index']:.2f}")
        st.markdown(f"**Cash left:** ${result['cash_left']:,.0f}")

        recs = result["recommendations"].copy()
        buy_count = (recs["action"] == "BUY ✅").sum()
        st.markdown(f"**Top‑K selected:** {buy_count} / {len(recs)}")
        st.markdown("**Guardrail:** only top 30% global tickers are eligible")
        if buy_count == 0:
            status_box("No eligible tickers under the guardrail for this selection.")
        recs = recs.rename(
            columns={
                "ticker": "Ticker",
                "action": "Action",
                "allocation_eur": "Allocation ($)",
                "score": "Score",
                "rank_pct": "Rank pct",
                "rank_pct_global": "Global rank pct",
            }
        )
        recs.index.name = "Rank"
        recs["Allocation ($)"] = recs["Allocation ($)"].round(2)

        rank_pct = recs["Global rank pct"] if "Global rank pct" in recs.columns else None
        recs_display = recs.drop(columns=["Rank pct", "Global rank pct"], errors="ignore")

        def _score_style(row):
            styles = [""] * len(row)
            if rank_pct is None or "Score" not in row.index:
                return styles
            pct = float(rank_pct.loc[row.name])
            if pct <= 0.3:
                color = "color: #2e7d32"
            elif pct >= 0.7:
                color = "color: #c62828"
            else:
                color = "color: #b26a00"
            styles[row.index.get_loc("Score")] = color
            return styles

        styled = (
            recs_display.style
            .format({"Allocation ($)": "${:,.2f}", "Score": "{:.4f}"})
            .apply(_score_style, axis=1)
        )
        st.dataframe(styled, use_container_width=True)
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
