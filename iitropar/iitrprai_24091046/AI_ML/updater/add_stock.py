import os
import pandas as pd

from config.config import MASTER_CSV
from data.data_loader import get_stock_data, load_master
from models.trainer import tune_and_train_base
from models.fine_tuner import fine_tune_stock

def add_new_stock_and_retrain_base(symbol, retrain_base=False, optuna_tune=False):
    """
    Add new stock to master; if retrain_base=True then:
      - append stock history to master
      - refit scaler on full master (required when adding very different stocks)
      - retrain base model (heavy)
    """
    master = load_master() or pd.DataFrame()
    new_stock = get_stock_data(symbol)
    master = pd.concat([master, new_stock], ignore_index=True).drop_duplicates(subset=['date','symbol']).reset_index(drop=True)
    master.to_csv(MASTER_CSV, index=False)
    print(f"Added {symbol} to master ({len(new_stock)} rows).")

    if retrain_base:
        print("Refitting scaler and retraining base model (this may take time)...")
        tune_and_train_base(master_df=master, n_trials=10 if optuna_tune else 0, optuna_tune=optuna_tune)
    else:
        fine_tune_stock(symbol, epochs=5)

    return True
