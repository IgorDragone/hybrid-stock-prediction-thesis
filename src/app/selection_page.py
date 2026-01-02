import streamlit as st

def traffic_light_animated(status="OFF"):
    colors = {
        "OFF": "#444444",
        "BUY": "#00C853",
        "HOLD": "#FFD600",
        "SELL": "#D50000"
    }

    color = colors.get(status, "#444444")

    st.markdown(
        f"""
        <style>
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 5px {color}; }}
            50% {{ box-shadow: 0 0 30px {color}; }}
            100% {{ box-shadow: 0 0 5px {color}; }}
        }}
        </style>

        <div style="
            width: 90px;
            height: 90px;
            margin: auto;
            border-radius: 50%;
            background-color: {color};
            animation: pulse 1.5s infinite;
        ">
        </div>
        """,
        unsafe_allow_html=True
    )

def render():
    st.title("Stock Analysis")

    st.markdown(
        """
        Welcome to the Stock Analysis Page! Here, you can explore various
        stock market data, visualize trends, and gain insights to make
        informed investment decisions. Use the tools and charts provided
        to analyze stock performance and identify potential opportunities.
        """
    )
    st.markdown( "### 📈 Graphics")

    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        stock = st.selectbox(
            "Choose a stock",
            ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        )

        indicator = st.selectbox(
            "Choose an indicator",
            ["Moving Average", "RSI", "MACD", "Bollinger Bands"]
        )

        time_horizon = st.pills(
            "Select Time Horizon",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"]
        )

    with col2:
        st.markdown(
            f"<div style='text-align: center;'> Stock Analysis for {stock}</div>",
            unsafe_allow_html=True
        )
        # Placeholder for stock analysis content
        st.line_chart({
            'Price': [150, 152, 153, 151, 155, 157, 160],
            'Volume': [2000, 2200, 2100, 2300, 2500, 2400, 2600]
        })

    st.divider()
    light_placeholder = st.empty()
    light_placeholder.markdown(f"### 🚦 Trading Signal for {stock}")
    traffic_placeholder = st.empty()
    signal = ""


    with traffic_placeholder:
        traffic_light_animated("OFF")

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
      if st.button("Generate Trading Signal", width="stretch"):
          import random
          signal = random.choice(["BUY", "HOLD", "SELL"])
          with traffic_placeholder:
              traffic_light_animated(signal)
    if signal:
      st.write(f"Giving these conditions, the recommended action is to {signal}")