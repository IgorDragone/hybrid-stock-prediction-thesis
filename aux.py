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
 
# we divide it in groups of 8
group = 8

group_1 = ["AMT", "CAT", "CVX", "GE", "GS", "NKE", "PFE", "UNH"]
group_2 = ["JPM", "BAC", "MS", "AXP", "MCD", "SBUX", "PEP"]
group_3 = ["PG", "WMT", "COST", "MRK", "ABBV", "COP", "SHEL"] 
group_4 = ["TTE", "BA", "MMM", "UNP", "NEE", "DUK", "SO", "D"]
group_5 = ["AEP", "PLD", "SPG", "PSA", "EQIX"]

# print("Group 1:", group_1)
print("Group 2:", group_2)
# print("Group 3:", group_3)
# print("Group 4:", group_4)
# print("Group 5:", group_5)

from src.data.config import AV_API_KEY, RAW_AV_DIR
from src.data.fundamentals_av import fetch_quarterly_fundamentals_av
for ticker in group_2:
    df_fund = fetch_quarterly_fundamentals_av(ticker=ticker, api_key=AV_API_KEY, cache_dir=RAW_AV_DIR)
# for ticker in tickers:
#     t = yf.Ticker(ticker)
#     info = t.info  # dict

#     sector = info.get("sector")
#     industry = info.get("industry")

#     print(f"{ticker}: {sector} | {industry}")
tickers_we_already_have = group_1 + group_2+ tickers_we_already_have
earliest_bal_date = {"date": None, "ticker": None}
earliest_inc_date = {"date": None, "ticker": None}
earliest_cf_date = {"date": None, "ticker": None}
for t in tickers_we_already_have:
    TICKER = t
    INC_RAW = RAW_AV_DIR  / "income" / f"{TICKER}.json"
    BAL_RAW = RAW_AV_DIR  / "balance" / f"{TICKER}.json"
    CF_RAW = RAW_AV_DIR  / "cashflow" / f"{TICKER}.json"
    with open(INC_RAW, "r") as f:
        data_inc = f.read()

    with open(BAL_RAW, "r") as f:
        data_bal = f.read()

    with open(CF_RAW, "r") as f:
        data_cf = f.read()

    # print the earlieast date available
    import json
    data_inc_json = json.loads(data_inc)
    quarterly_reports = data_inc_json.get("quarterlyReports", [])
    earliest_report = quarterly_reports[-1]
    earliest_balance_date = earliest_report.get("fiscalDateEnding")
    if earliest_inc_date["date"] is None or earliest_balance_date < earliest_inc_date["date"]:
        earliest_inc_date["date"] = earliest_balance_date
        earliest_inc_date["ticker"] = TICKER

    data_bal_json = json.loads(data_bal)
    quarterly_reports_bal = data_bal_json.get("quarterlyReports", [])
    earliest_report_bal = quarterly_reports_bal[-1]
    earliest_balance_date_bal = earliest_report_bal.get("fiscalDateEnding")
    if earliest_bal_date["date"] is None or earliest_balance_date_bal < earliest_bal_date["date"]:
        earliest_bal_date["date"] = earliest_balance_date_bal
        earliest_bal_date["ticker"] = TICKER
    

    data_cf_json = json.loads(data_cf)
    quarterly_reports_cf = data_cf_json.get("quarterlyReports", [])
    earliest_report_cf = quarterly_reports_cf[-1]
    earliest_balance_date_cf = earliest_report_cf.get("fiscalDateEnding")
    if earliest_cf_date["date"] is None or earliest_balance_date_cf < earliest_cf_date["date"]:
        earliest_cf_date["date"] = earliest_balance_date_cf
        earliest_cf_date["ticker"] = TICKER

print("Earliest Income Statement date:", earliest_inc_date["date"], "Ticker:", earliest_inc_date["ticker"])
print("Earliest Balance Sheet date:", earliest_bal_date["date"], "Ticker:", earliest_bal_date["ticker"])
print("Earliest Cash Flow date:", earliest_cf_date["date"], "Ticker:", earliest_cf_date["ticker"])
    
