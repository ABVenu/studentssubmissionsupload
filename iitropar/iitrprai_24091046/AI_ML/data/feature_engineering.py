import pandas as pd
import ta

# Normalize DataFrame column names to lowercase (handles MultiIndex from yfinance).
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['BB_high'] = ta.volatility.bollinger_hband(df['close'], window=20)
    df['BB_low'] = ta.volatility.bollinger_lband(df['close'], window=20)
    df['STOCH'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['momentum'] = df['close'] - df['close'].shift(1)
    df['volatility'] = df['close'].rolling(10).std()
    df['roc'] = ta.momentum.roc(df['close'], window=5)
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    return df
