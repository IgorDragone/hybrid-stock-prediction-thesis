import streamlit as st


def render(mode: str):
    st.session_state.db_mode = st.radio(
        "Choose database path",
        ["Load existing database", "Create new database"],
        index=0 if mode == "Load existing database" else 1,
    )

    if st.session_state.db_mode == "Load existing database":
        db_list = [
            "financial_database_preprocessed.parquet",
            "financial_database_features.parquet",
            "financial_database_model_ready.parquet",
        ]
        st.session_state.db_candidate = st.selectbox("Available databases", db_list)
        if st.button("Confirm database", use_container_width=True):
            st.session_state.db_selected = st.session_state.db_candidate
            st.session_state.db_confirmed = True
    else:
        st.session_state.db_params["horizon"] = st.selectbox(
            "Time horizon",
            ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
            index=1,
        )
        st.session_state.db_params["tickers"] = st.slider(
            "Number of tickers",
            min_value=5,
            max_value=100,
            value=10,
            step=5,
        )
        if st.button("Confirm database config", use_container_width=True):
            st.session_state.db_selected = (
                f"Custom ({st.session_state.db_params['horizon']}, "
                f"{st.session_state.db_params['tickers']} tickers)"
            )
            st.session_state.db_confirmed = True
