import os
from datetime import datetime
import pandas as pd
import yfinance as yf

from config.config import START_DATE, END_DATE, MASTER_CSV
from .feature_engineering import _normalize_columns, compute_technical_indicators

def get_stock_data(symbol, start=START_DATE, end=END_DATE):
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {symbol} between {start} and {end}")
    df = df.reset_index()
    df = _normalize_columns(df)
    df = compute_technical_indicators(df)
    df['symbol'] = symbol
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df.dropna().reset_index(drop=True)

def build_master_dataset(symbols, start=START_DATE, end=END_DATE, save=True):
    dfs = []
    for s in symbols:
        print(f"Downloading {s} ...")
        dfs.append(get_stock_data(s, start=start, end=end))
    master = pd.concat(dfs, ignore_index=True)
    if save:
        master.to_csv(MASTER_CSV, index=False)
        print(f"Saved master CSV -> {MASTER_CSV}, rows={len(master)}")
    return master

def load_master():
    if not os.path.exists(MASTER_CSV):
        return None
    m = pd.read_csv(MASTER_CSV)
    if 'date' in m.columns:
        m['date'] = pd.to_datetime(m['date'])
    return m
