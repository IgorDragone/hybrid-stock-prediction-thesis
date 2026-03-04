import streamlit as st


def render(mode: str):
    if "model_mode_prev" not in st.session_state:
        st.session_state.model_mode_prev = mode
    if st.session_state.model_mode_prev != mode and mode == "Create new model":
        st.session_state.db_selected = None
        st.session_state.model_pending = None
    st.session_state.model_mode_prev = mode

    if mode == "Load existing model":
        model_list = [m for m in st.session_state.available_models if m != "buy_hold_eqw"]
        if not model_list:
            st.warning("No saved models found.")
            return
        label_map = {
            "baseline_mom": "Momentum Baseline",
            "ridge": "Ridge (Linear)",
            "hgb": "Gradient Boosting (HGB)",
        }
        display = {mid: label_map.get(mid, mid) for mid in model_list}
        st.session_state.model_candidate = st.selectbox(
            "Available models",
            model_list,
            format_func=lambda mid: display.get(mid, mid),
        )
        if st.button("Use this model", use_container_width=True, key="use_model_existing"):
            st.session_state.model_selected = st.session_state.model_candidate
            st.session_state.db_selected = st.session_state.model_db_map.get(
                st.session_state.model_selected
            )
            st.session_state.ui_page = "summary_page"
    else:
        st.info("Model creation is temporarily disabled. Use an existing model.")
        return
        st.session_state.model_type = st.selectbox(
            "Model type",
            ["XGBoost", "Random Forest", "LSTM", "Transformer"],
            index=0,
        )
        if "model_type_prev" not in st.session_state:
            st.session_state.model_type_prev = st.session_state.model_type
        if st.session_state.model_type_prev != st.session_state.model_type:
            st.session_state.db_selected = None
            st.session_state.model_pending = None
        st.session_state.model_type_prev = st.session_state.model_type
        if st.button("Continue to Database", use_container_width=True, key="use_model_new"):
            st.session_state.model_pending = st.session_state.model_type
            st.session_state.ui_page = "db_page"
