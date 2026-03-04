import streamlit as st


def apply_style():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Libre Baskerville", "Georgia", serif;
        }
        h1, h2, h3, h4, h5 {
            font-family: "Montserrat", "Helvetica", sans-serif;
        }
        .flow-card {
            background: #f8f6f2;
            border: 1px solid #e6e6e6;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        }
        .flow-title {
            font-size: 22px;
            font-weight: 600;
            color: #111111;
            margin: 0 0 8px 0;
        }
        .flow-sub {
            color: #3f3f3f;
            margin: 0 0 8px 0;
        }
        .status-box {
            background: #f2f2f2;
            border: 1px solid #d4d4d4;
            border-radius: 14px;
            padding: 12px 16px;
            color: #2b2b2b;
            margin: 8px 0 18px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.markdown(f"<p style='color:#6b6b6b'>{subtitle}</p>", unsafe_allow_html=True)


def status_box(message: str):
    st.markdown(
        f"<div class='status-box'>{message}</div>",
        unsafe_allow_html=True,
    )
