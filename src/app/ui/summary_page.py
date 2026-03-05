import streamlit as st
from uuid import uuid4

from src.app.logic.watchlists import save_watchlists
from src.app.logic.recommendations import score_watchlist

from .ui_components import header, status_box

MAX_PORTFOLIOS = 3


def summary_page(summary_fn):
    header(
        "Recommendation Summary",
        "Review the watchlist inputs and model-driven decisions before saving.",
    )
    summary_fn(show_model=True)

    st.subheader("Decisions")
    try:
        if not st.session_state.model_selected:
            status_box("Select a model to generate recommendations.")
            return
        result = score_watchlist(
            model_id=st.session_state.model_selected,
            tickers=st.session_state.watchlist_tickers,
        )
        if result.get("used_fallback"):
            status_box("Model could not be loaded (joblib missing). Using baseline score instead.")
        st.markdown(f"**Exposure:** {result['exposure']:.0%}")
        if result["exposure"] < 1.0:
            st.markdown("**Positioning:** Defensive (risk-off)")
        else:
            st.markdown("**Positioning:** Full risk-on")
        if result.get("stress_index") is not None:
            st.markdown(f"**Stress index:** {result['stress_index']:.2f}")
        st.markdown("**Macro overlay:** Exposure is reduced when stress index is above threshold.")

        recs = result["recommendations"].copy()
        universe_size = int(result.get("universe_size", len(recs)))
        buy_count = int((recs["action"] == "BUY ✅").sum())
        sell_count = int((recs["action"] == "SELL ⛔").sum())
        hold_count = int((recs["action"] == "HOLD ⏸️").sum())
        st.markdown(f"**Actions:** {buy_count} BUY / {hold_count} HOLD / {sell_count} SELL")
        st.markdown("**Rule:** BUY = top 30% global · SELL = bottom 30% global · HOLD = middle 40%")
        if buy_count == 0:
            status_box("No BUY signals in this watchlist under the global ranking rule.")
        def _global_rank_label(row):
            rank_global = int(row["rank_global"])
            top_pct = max(1, int(round(100 * rank_global / universe_size)))
            bottom_pct = max(1, int(round(100 * (universe_size - rank_global + 1) / universe_size)))
            side = "top" if rank_global <= universe_size / 2 else "bottom"
            pct = top_pct if side == "top" else bottom_pct
            return f"{rank_global} / {universe_size} - {side} {pct}%"

        recs["global_rank_label"] = recs.apply(_global_rank_label, axis=1)
        recs = recs.rename(
            columns={
                "ticker": "Ticker",
                "company": "Company",
                "sector": "Sector",
                "action": "Action",
                "recommendation_level": "Recommendation level",
                "global_rank_label": "Global rank",
                "rank_pct": "Rank pct",
                "rank_pct_global": "Global rank pct",
            }
        )
        recs.index.name = "Rank"

        recs_display = recs.drop(columns=["rank_global", "Rank pct", "Global rank pct"], errors="ignore")
        recs_display = recs_display[
            [
                "Ticker",
                "Company",
                "Sector",
                "Action",
                "Recommendation level",
                "Global rank",
            ]
        ]
        def _level_style(row):
            styles = [""] * len(row)
            if "Recommendation level" not in row.index:
                return styles
            level = str(row["Recommendation level"])
            color_map = {
                "Very High": "color: #1b5e20; font-weight: 700",
                "High": "color: #43a047; font-weight: 600",
                "Medium (Upper)": "color: #f9a825; font-weight: 600",
                "Medium (Lower)": "color: #b26a00; font-weight: 600",
                "Low": "color: #e53935; font-weight: 600",
                "Very Low": "color: #8e0000; font-weight: 700",
            }
            styles[row.index.get_loc("Recommendation level")] = color_map.get(level, "")
            return styles

        styled = recs_display.style.apply(_level_style, axis=1)
        st.dataframe(styled, width="stretch")
    except Exception as exc:  # noqa: BLE001
        status_box(f"Unable to compute recommendations: {exc}")

    st.subheader("Save Watchlist")
    if len(st.session_state.saved_watchlists) >= MAX_PORTFOLIOS:
        existing_ids = {p.get("id") for p in st.session_state.saved_watchlists}
        current_id = st.session_state.get("watchlist_id")
        if current_id not in existing_ids:
            status_box("Maximum watchlists reached. Delete one to save a new watchlist.")
            st.button("Save Watchlist", width="stretch", disabled=True)
            return

    if st.button("Save Watchlist", width="stretch"):
        current_id = st.session_state.get("watchlist_id")
        if not current_id:
            current_id = str(uuid4())
            st.session_state.watchlist_id = current_id
        existing = None
        for p in st.session_state.saved_watchlists:
            if p.get("id") == current_id:
                existing = p
                break
        if existing is None:
            # Backward-compatible fallback for old saved watchlists without id.
            for p in st.session_state.saved_watchlists:
                if p.get("name") == st.session_state.watchlist_name:
                    existing = p
                    break
        payload = {
            "id": current_id,
            "name": st.session_state.watchlist_name,
            "tickers": st.session_state.watchlist_tickers,
            "model": st.session_state.model_selected,
            "database": st.session_state.db_selected,
        }
        if existing:
            existing.update(payload)
        else:
            st.session_state.saved_watchlists.append(payload)
        save_watchlists(st.session_state.saved_watchlists)
        st.success("Watchlist saved.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Models", width="stretch"):
            st.session_state.ui_page = "model_page"
    with col2:
        if st.button("Back to Home", width="stretch"):
            st.session_state.ui_page = "watchlist_home"
