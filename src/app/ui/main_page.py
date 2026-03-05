from pathlib import Path

import streamlit as st

ASSETS_PATH = Path(__file__).resolve().parents[1] / "assets"

def render():

    # TITOLO
    st.markdown(
        "<h1 style='text-align: center;'> ☯︎ Dragon Trading ☯︎</h1>",
        unsafe_allow_html=True
    )

    # SOTTOTITOLO
    st.markdown(
        "<h4 style='text-align: center; color: gray;'>"
        "AI-powered Investment Recommendation System"
        "</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    # HERO IMAGE
    st.image(
        str(ASSETS_PATH / "dragon_trading3.png"),
        caption="Turning data into investment insights"
    )

    st.write("")

    # DESCRIZIONE
    st.markdown(
        """
        **Dragon Trading** is an innovative platform that leverages
        cutting-edge AI technology to provide personalized stock investment
        recommendations. Whether you're a seasoned investor or just starting
        out, our system analyzes market trends and your risk profile to help
        you make informed investment decisions. Join us on a journey to smarter investing
        and unlock your financial potential with Dragon Trading.
        """
    )

    st.write("")

    # BOTTONE
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("START NOW", width="stretch"):
            st.session_state.page = 2
