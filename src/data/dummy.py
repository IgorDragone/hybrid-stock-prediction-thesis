from fundamentals import fetch_quarterly_fundamentals
import yfinance as yf
from config import START_DATE, END_DATE, FRED_SERIES
from macro import fetch_macro_fred

# Fetch macroeconomic data
macro_data = fetch_macro_fred(FRED_SERIES, START_DATE, END_DATE)
#we print the GDP series value without missing dates
print(macro_data)
# we print the row of 31st december 2

#fetch_quarterly_fundamentals(ticker)

# # src/data/fundamentals.py
# import requests
# import pandas as pd


# def merge_asof(left, right):
#     return pd.merge_asof(
#         left.sort_values('date'),
#         right.sort_values('date'),
#         on='date',
#         direction='backward'
#     )

# def fetch_fundamentals_fmp(ticker, api_key):
#     url = (
#         f"https://financialmodelingprep.com/stable/search-exchange-variants?symbol={ticker}&apikey={api_key}"
#     )

#     response = requests.get(url)
#     print(response.status_code)
#     print(response.text)
#     # data = requests.get(url).json()
#     # if not data:
#     #     raise ValueError(f"No fundamentals for {ticker}")

#     # if isinstance(data, dict):
#     #     data = [data]
    
#     # print(data)  # Debugging line to inspect the fetched data
#     exit()

#     df = pd.DataFrame(data)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values('date')
#     exit()

#     cols = {
#         'peRatio': 'pe',
#         'pbRatio': 'pb',
#         'roe': 'roe',
#         'roa': 'roa',
#         'netProfitMargin': 'net_margin',
#         'operatingMargin': 'op_margin',
#         'revenueGrowth': 'revenue_growth',
#         'earningsGrowth': 'earnings_growth',
#         'debtToEquity': 'debt_to_equity',
#         'currentRatio': 'current_ratio'
#     }

#     df = df[['date'] + list(cols.keys())]
#     df = df.rename(columns=cols)

#     return df


# TICKERS = [
#     "AAPL", "MSFT", "GOOGL", "AMZN",
#     "META", "NVDA", "TSLA",
#     "JPM", "JNJ", "XOM", "KO"
# ]

# #for every ticker, we check which fundamentals are common to all tickers

# import yfinance as yf

# for ticker in TICKERS:
#     print(f"Fetching fundamentals for {ticker}")
#     tk = yf.Ticker(ticker)

#     fundamentals = tk.info.keys()
#     if 'common_fundamentals' not in locals():
#       common_fundamentals = set(fundamentals)
#     else:
#       common_fundamentals.intersection_update(fundamentals)
#     # print(f"Number of fundamentals for {ticker}: {len(fundamentals)}")
#     # print("Number of common fundamentals so far:", len(common_fundamentals))

# # print("Common fundamentals across all tickers:")
# sorted_common = sorted(common_fundamentals)
# # for fundamental in sorted_common:
# #     print(fundamental)

# # we send it to a .txt file
# with open("common_fundamentals.txt", "w") as f:
#     for fundamental in sorted_common:
#         f.write(fundamental + "\n")

