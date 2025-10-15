import os
import json

stock_list = ['SBIN.NS', 'TCS.NS', 'RELIANCE.NS']
START_DATE = '2001-02-15'
END_DATE = '2025-07-25'

# Folder structure (created dynamically)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_store") # separate place to save artifacts
ARTIFACTS_DIR = MODELS_DIR

# Create folders dynamically
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Paths to artifacts
BASE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "base_model.h5")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
PARAMS_PATH = os.path.join(ARTIFACTS_DIR, "params.json")
MASTER_CSV = os.path.join(ARTIFACTS_DIR, "master_data.csv")

# Features + defaults
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'volatility',
            'roc', 'williams_r', 'SMA_20', 'EMA_20', 'MACD', 'RSI',
            'BB_high', 'BB_low', 'STOCH', 'ATR']

DEFAULT_PARAMS = {'units': 128, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 32, 'seq_len': 60}

def load_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, 'r') as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()

def save_params(params: dict):
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f)
