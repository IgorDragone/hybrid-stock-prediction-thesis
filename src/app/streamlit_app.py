import logging

import main_page
import selection_page
import streamlit as st

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger.info("Starting Streamlit app")

st.set_page_config(
    page_title="Dragon Trading",
    page_icon="🐉",
    layout="wide"
)

# inizializza stato
if "page" not in st.session_state:
    st.session_state.page = 1

# routing
if st.session_state.page == 1:
    main_page.render()
elif st.session_state.page == 2:
    selection_page.render()
