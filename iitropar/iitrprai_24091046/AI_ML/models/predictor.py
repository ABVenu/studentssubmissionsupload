import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from config.config import (
    FEATURES, MASTER_CSV, BASE_MODEL_PATH, PARAMS_PATH, DEFAULT_PARAMS, SCALER_PATH
)
from data.data_loader import load_master
from data.preprocessing import load_scaler

def forecast_next_days(symbol, N=5, use_finetuned=True):
    master = load_master()
    if master is None:
        raise FileNotFoundError("Master CSV not found. Train base model first.")

    stock_model_path = os.path.join(os.path.dirname(BASE_MODEL_PATH), f"{symbol.replace('.','_')}_ft.h5")
    if use_finetuned and os.path.exists(stock_model_path):
        model = load_model(stock_model_path)
        print(f"Using fine-tuned model for {symbol}")
    else:
        model = load_model(BASE_MODEL_PATH)
        print(f"Using base model for {symbol}")

    params = json.load(open(PARAMS_PATH)) if os.path.exists(PARAMS_PATH) else DEFAULT_PARAMS
    seq_len = params['seq_len']

    stock_df = master[master['symbol'] == symbol].sort_values(by='date').reset_index(drop=True)
    if stock_df.empty:
        raise ValueError(f"No data for {symbol} in master dataset")

    scaler = load_scaler()
    stock_scaled = stock_df.copy()
    stock_scaled[FEATURES] = scaler.transform(stock_scaled[FEATURES])
    last_sequence = stock_scaled[FEATURES].values[-seq_len:]
    forecast = []
    current_input = last_sequence.copy()

    for _ in range(N):
        inp = current_input.reshape(1, seq_len, len(FEATURES))
        pred_scaled = model.predict(inp, verbose=0)[0, 0]
        pred_full = np.zeros((len(FEATURES),))
        pred_full[FEATURES.index('close')] = pred_scaled
        unscaled_pred = scaler.inverse_transform([pred_full])[0][FEATURES.index('close')]
        forecast.append(unscaled_pred)

        new_row = current_input[-1].copy()
        new_row[FEATURES.index('close')] = pred_scaled
        current_input = np.vstack((current_input[1:], new_row))

    last_date = stock_df['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=N, freq='B')

    plt.figure(figsize=(8, 4))
    plt.plot(forecast_dates, forecast, marker='o', linestyle='--')
    plt.title(f"{N}-day Forecast for {symbol} (use_finetuned={use_finetuned})")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Close")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return list(zip(forecast_dates, forecast))
