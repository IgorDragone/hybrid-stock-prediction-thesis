import streamlit as st

from ui_components import status_box


def render(mode: str):
    if "model_mode_prev" not in st.session_state:
        st.session_state.model_mode_prev = mode
    if st.session_state.model_mode_prev != mode and mode == "Create new model":
        st.session_state.db_selected = None
        st.session_state.model_pending = None
    st.session_state.model_mode_prev = mode

    if mode == "Load existing model":
        model_list = st.session_state.available_models
        st.session_state.model_candidate = st.selectbox(
            "Available models",
            model_list,
        )
        if st.button("Use this model", use_container_width=True, key="use_model_existing"):
            st.session_state.model_selected = st.session_state.model_candidate
            st.session_state.db_selected = st.session_state.model_db_map.get(
                st.session_state.model_selected
            )
            st.session_state.ui_page = "summary_page"
    else:
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
