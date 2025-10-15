import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config.config import SCALER_PATH, FEATURES

def fit_and_save_scaler(master_df: pd.DataFrame):
    """
    Fit MinMaxScaler on FEATURES and save to disk.
    Returns: scaler, scaled DataFrame
    """
    scaler = MinMaxScaler()
    df_copy = master_df.copy()
    df_copy[FEATURES] = scaler.fit_transform(df_copy[FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved to {SCALER_PATH}")
    return scaler, df_copy

def load_scaler():
    """
    Load previously saved scaler.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found - train base model first.")
    return joblib.load(SCALER_PATH)

def create_sequences_from_values(values: np.ndarray, seq_len: int):
    """
    Create sequence samples for time series forecasting.
    """
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i, FEATURES.index('close')])
    return np.array(X), np.array(y)

def create_sequences_multi(df: pd.DataFrame, seq_len: int):
    """
    Create sequences for all symbols in a multi-stock DataFrame.
    """
    X, y = [], []
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].sort_values(by='date').reset_index(drop=True)
        vals = stock_df[FEATURES].values
        x_sym, y_sym = create_sequences_from_values(vals, seq_len)
        if len(x_sym):
            X.append(x_sym)
            y.append(y_sym)
    if len(X) == 0:
        return np.empty((0, seq_len, len(FEATURES))), np.empty((0,))
    return np.vstack(X), np.hstack(y)
