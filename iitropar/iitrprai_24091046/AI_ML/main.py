from config.config import stock_list, SCALER_PATH, BASE_MODEL_PATH
from data.data_loader import build_master_dataset, load_master
from data.preprocessing import fit_and_save_scaler, load_scaler
from models.trainer import tune_and_train_base
from models.fine_tuner import fine_tune_stock
from models.predictor import forecast_next_days
import os

if __name__ == "__main__":
    if not os.path.exists(load_master.__globals__['MASTER_CSV']):
        master = build_master_dataset(stock_list)
    else:
        master = load_master()

    if not os.path.exists(SCALER_PATH):
        scaler, master_scaled = fit_and_save_scaler(master)
    else:
        scaler = load_scaler()

    if not os.path.exists(BASE_MODEL_PATH):
        base_model, params = tune_and_train_base(master, n_trials=10, optuna_tune=False)
    else:
        print("Base model found, skipping full training.")

    for s in stock_list:
        fine_tune_stock(s, epochs=5)

    print("Forecast for SBIN.NS (using fine-tuned):")
    fc = forecast_next_days('SBIN.NS', N=5, use_finetuned=True)
    for d, p in fc:
        print(d.date(), round(p, 2))
