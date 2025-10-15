import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

from config.config import (
    BASE_MODEL_PATH, SCALER_PATH, PARAMS_PATH, DEFAULT_PARAMS, FEATURES, MODELS_DIR
)
from data.data_loader import get_stock_data, load_master
from data.preprocessing import load_scaler, create_sequences_from_values

def fine_tune_stock(symbol, epochs=5, lr_override=None, seq_len_override=None, save=True):
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError("Base model not found. Train base model first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Fit scaler first.")

    params = json.load(open(PARAMS_PATH)) if os.path.exists(PARAMS_PATH) else DEFAULT_PARAMS
    seq_len = seq_len_override or params['seq_len']

    base_model = load_model(BASE_MODEL_PATH, compile=False)

    use_lr = (lr_override if lr_override is not None else max(1e-5, params['lr'] / 5))
    base_model.compile(optimizer=Adam(learning_rate=use_lr), loss=MeanSquaredError())

    master = load_master()
    if master is None or symbol not in master['symbol'].unique():
        stock_df = get_stock_data(symbol)
    else:
        stock_df = master[master['symbol'] == symbol].sort_values(by='date').reset_index(drop=True)

    scaler = load_scaler()
    stock_df_scaled = stock_df.copy()
    stock_df_scaled[FEATURES] = scaler.transform(stock_df_scaled[FEATURES])

    X_stock, y_stock = create_sequences_from_values(stock_df_scaled[FEATURES].values, seq_len)
    if len(X_stock) == 0:
        print(f"Not enough data to fine-tune for {symbol}")
        return None

    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    base_model.fit(X_stock, y_stock, epochs=epochs, batch_size=max(1, len(X_stock)//10), callbacks=[es], verbose=1)

    stock_model_path = os.path.join(MODELS_DIR, f"{symbol.replace('.','_')}_ft.h5")
    if save:
        base_model.save(stock_model_path)
        print(f"Saved fine-tuned model for {symbol} -> {stock_model_path}")
    return base_model
