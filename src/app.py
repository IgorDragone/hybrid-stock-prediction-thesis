import streamlit as st

st.set_page_config(page_title="Investment Recommender", layout="centered")

# inizializzazione stato
if "page" not in st.session_state:
    st.session_state.page = 1

# ---------- PAGE 1: HOME ----------
if st.session_state.page == 1:
    st.title("📈 Investment Recommendation System")
    st.write("""
    Questo sistema fornisce una raccomandazione di investimento
    basata su dati storici e indicatori finanziari.
    """)
    
    if st.button("Inizia"):
        st.session_state.page = 2

elif st.session_state.page == 2:
    st.title("🔍 Seleziona il titolo")

    stock = st.selectbox(
        "Scegli un titolo azionario",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    )

    risk = st.selectbox(
        "Profilo di rischio",
        ["Basso", "Medio", "Alto"]
    )

    if st.button("Ottieni raccomandazione"):
        st.session_state.stock = stock
        st.session_state.risk = risk
        st.session_state.page = 3