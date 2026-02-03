import yfinance as yf

"""
1. Technology
- Apple
- Microsoft
- NVIDIA
- Alphabet
- Meta Platforms
2. Financials
- JPMorgan Chase
- Bank of America
- Goldman Sachs
- Morgan Stanley
- American Express
3. Consumer Discretionary
- Amazon
- Tesla
- Nike
- McDonald's
- Starbucks
4. Consumer Staples
- Coca-Cola
- PepsiCo
- Procter & Gamble
- Walmart
- Costco
5. Healthcare
- Johnson & Johnson
- Pfizer
- Merck
- UnitedHealth Group
- AbbVie
6. Energy
- Exxon Mobil
- Chevron
- ConocoPhillips
- Shell
- TotalEnergies
7. Industrials
- Boeing
- Caterpillar
- General Electric
- 3M
- Union Pacific
8. Utilities
- NextEra Energy
- Duke Energy
- Southern Company
- Dominion Energy
- American Electric Power
9. Real Estate
- Prologis
- American Tower
- Simon Property Group
- Public Storage
- Equinix

"""

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "JPM", "BAC", "GS", "MS", "AXP", "AMZN", "TSLA", "NKE", "MCD", "SBUX"
           , "KO", "PEP", "PG", "WMT", "COST", "JNJ", "PFE", "MRK", "UNH", "ABBV", "XOM", "CVX", "COP", "SHEL", "TTE",
           "BA", "CAT", "GE", "MMM", "UNP", "NEE", "DUK", "SO", "D", "AEP", "PLD", "AMT", "SPG", "PSA", "EQIX"]

tickers_we_already_have = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Non-tech
    "KO", "JNJ", "XOM",
]

tickers_we_want = list(set(tickers) - set(tickers_we_already_have))
print("Tickers we want to fetch info for:", tickers_we_want)
print("Number of tickers to fetch info for:", len(tickers_we_want))

# we divide it in groups of 8
group = 8

group_1 = tickers_we_want[0:group]
group_2 = tickers_we_want[group:2*group]
group_3 = tickers_we_want[2*group:3*group]
group_4 = tickers_we_want[3*group:4*group]
group_5 = tickers_we_want[4*group:5*group]

print("Group 1:", group_1)
print("Group 2:", group_2)
print("Group 3:", group_3)
print("Group 4:", group_4)
print("Group 5:", group_5)

from src.data.config import AV_API_KEY, RAW_AV_DIR
from src.data.fundamentals_av import fetch_quarterly_fundamentals_av
for ticker in group_1:
    df_fund = fetch_quarterly_fundamentals_av(ticker=ticker, api_key=AV_API_KEY, cache_dir=RAW_AV_DIR)
# for ticker in tickers:
#     t = yf.Ticker(ticker)
#     info = t.info  # dict

#     sector = info.get("sector")
#     industry = info.get("industry")

#     print(f"{ticker}: {sector} | {industry}")