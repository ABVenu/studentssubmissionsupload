# backend/api/predict_api.py  (replace the route with this version)

import os, json
import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from tensorflow.keras.models import load_model

router = APIRouter()

_prediction_cache = {}

# Paths
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(_CURRENT_DIR, "..", ".."))

default_ai_ml_dir = os.path.join(project_root, "AI_ML")
DEFAULT_MODEL_DIR = os.path.join(default_ai_ml_dir, "models_store")

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
DATA_PATH = os.path.join(MODEL_DIR, "master_data.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PARAMS_PATH = os.path.join(MODEL_DIR, "params.json")

# MUST match your training code exactly (case-sensitive)
FEATURES = [
    'open','high','low','close','volume','momentum','volatility',
    'roc','williams_r','SMA_20','EMA_20','MACD','RSI',
    'BB_high','BB_low','STOCH','ATR'
]

def _load_seq_len():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f).get("seq_len", 60)
    return 60

@router.get("/predict/{stock}")
def predict_stock(stock: str, n_days: int = 5):
    symbol = stock.strip().upper()

    # 1) Load scaler
    if not os.path.exists(SCALER_PATH):
        raise HTTPException(404, "scaler.pkl not found.")
    scaler = joblib.load(SCALER_PATH)

    # 2) Load master_data.csv (already contains indicators from training)
    if not os.path.exists(DATA_PATH):
        raise HTTPException(404, "master_data.csv not found.")
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    if 'symbol' not in df.columns:
        raise HTTPException(422, "master_data.csv must contain 'symbol' column as produced by the training script.")

    # filter & sort
    df_sym = df[df['symbol'].str.upper() == symbol].sort_values('date')
    if df_sym.empty:
        raise HTTPException(422, f"No rows for {symbol} in master_data.csv")

    # 3) seq_len from params.json (not hardcoded)
    seq_len = _load_seq_len()
    if len(df_sym) < seq_len:
        raise HTTPException(422, f"Need >= {seq_len} rows for {symbol}; have {len(df_sym)}.")

    # 4) Pick model: prefer fine-tuned; try multiple file-naming patterns; else base
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('.','_')}_ft.h5")
    if not os.path.exists(model_path):
        raise HTTPException(404, "No model file found (fine-tuned or base).")
    model = load_model(model_path)

    # 5) Validate features and order them EXACTLY as training
    missing = [c for c in FEATURES if c not in df_sym.columns]
    if missing:
        raise HTTPException(422, f"master_data.csv missing features {missing} (names/case must match training).")

    feats = df_sym[FEATURES].values
    scaled = scaler.transform(feats)           # shape [T, F]
    cur = scaled[-seq_len:, :].copy()          # last seq_len rows
    close_idx = FEATURES.index('close')

    # 6) Rolling 5-day prediction with proper inverse_transform
    preds = []
    for _ in range(n_days):
        yhat_scaled = float(model.predict(cur[np.newaxis, ...], verbose=0)[0, 0])

        # place predicted scaled close at the correct index, then inverse_transform
        pad = np.zeros((len(FEATURES),), dtype=np.float32)
        pad[close_idx] = yhat_scaled
        unscaled_close = scaler.inverse_transform(pad.reshape(1, -1))[0, close_idx]
        preds.append(unscaled_close)

        # roll window by replacing only the close in the last row
        new_row = cur[-1].copy()
        new_row[close_idx] = yhat_scaled
        cur = np.vstack([cur[1:], new_row])

    last_date = pd.to_datetime(df_sym['date'].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_days)

    return {
        "stock": symbol,
        "predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(float(p), 2)}
            for d, p in zip(future_dates, preds)
        ],
    }
