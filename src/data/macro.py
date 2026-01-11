# src/data/macro.py
import pandas as pd
from pandas_datareader import data as web

def fetch_macro_fred(series: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED for given series between start and end dates.

    Args:
        series (list): List of FRED series IDs to fetch.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing the requested FRED series with datetime index.
    """
    frames = []
    for s in series:
        ser = web.DataReader(s, 'fred', start, end)
        ser.columns = [s]
        frames.append(ser)

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    print(macro)
    return macro