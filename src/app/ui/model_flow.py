import streamlit as st


def render():
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
    default_model = st.session_state.get("model_selected")
    if default_model in model_list:
        default_index = model_list.index(default_model)
    else:
        default_index = 0
    st.session_state.model_candidate = st.selectbox(
        "Available models",
        model_list,
        index=default_index,
        format_func=lambda mid: display.get(mid, mid),
    )
    if st.button("Use this model", width="stretch", key="use_model_existing"):
        st.session_state.model_selected = st.session_state.model_candidate
        st.session_state.ui_page = "summary_page"
