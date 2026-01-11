# src/data/macro.py
import pandas as pd
from pandas_datareader import data as web

def fetch_macro_fred(series, start, end):
    frames = []
    for s in series:
        ser = web.DataReader(s, 'fred', start, end)
        ser.columns = [s]
        frames.append(ser)

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    return macro