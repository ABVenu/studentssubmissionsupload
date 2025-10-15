import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import optuna

from config.config import (
    FEATURES, BASE_MODEL_PATH, PARAMS_PATH, DEFAULT_PARAMS, MASTER_CSV, save_params
)
from data.preprocessing import fit_and_save_scaler, create_sequences_multi
from config.config import FEATURES
from models.model_builder import build_model

def tune_and_train_base(master_df, n_trials=10, optuna_tune=True):
    # scaler fit (and persisted)
    scaler, master_scaled = fit_and_save_scaler(master_df)

    if optuna_tune:
        def objective(trial):
            units = trial.suggest_int('units', 32, 256)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            seq_len = trial.suggest_int('seq_len', 30, 90)

            X, y = create_sequences_multi(master_scaled, seq_len)
            if len(X) == 0:
                return np.inf
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_model(units, dropout, lr, seq_len)
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=batch_size, callbacks=[es], verbose=0)

            preds = model.predict(X_test, verbose=0)
            pad = np.zeros((preds.shape[0], len(FEATURES)))
            pad[:, FEATURES.index('close')] = preds[:, 0]
            pred_prices = scaler.inverse_transform(pad)[:, FEATURES.index('close')]

            pad_y = np.zeros((y_test.shape[0], len(FEATURES)))
            pad_y[:, FEATURES.index('close')] = y_test
            actual_prices = scaler.inverse_transform(pad_y)[:, FEATURES.index('close')]

            rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial.params
        params = {
            'units': best['units'],
            'dropout': best['dropout'],
            'lr': best['lr'],
            'batch_size': best['batch_size'],
            'seq_len': best['seq_len']
        }
        print("Optuna best params:", params)
    else:
        params = DEFAULT_PARAMS.copy()
        print("Using default params:", params)

    save_params(params)

    seq_len = params['seq_len']
    X, y = create_sequences_multi(master_scaled, seq_len)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    base_model = build_model(params['units'], params['dropout'], params['lr'], seq_len)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    base_model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=params['batch_size'], callbacks=[es])

    base_model.save(BASE_MODEL_PATH)
    print("Saved base model ->", BASE_MODEL_PATH)

    preds = base_model.predict(X_test)
    pad = np.zeros((preds.shape[0], len(FEATURES)))
    pad[:, FEATURES.index('close')] = preds[:, 0]
    pred_prices = scaler.inverse_transform(pad)[:, FEATURES.index('close')]

    pad_y = np.zeros((y_test.shape[0], len(FEATURES)))
    pad_y[:, FEATURES.index('close')] = y_test
    actual_prices = scaler.inverse_transform(pad_y)[:, FEATURES.index('close')]

    print("Base model evaluation:")
    print("RMSE:", np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    print("MAE:", mean_absolute_error(actual_prices, pred_prices))
    print("R2:", r2_score(actual_prices, pred_prices))

    master_df.to_csv(MASTER_CSV, index=False)
    print("Saved master csv.")

    return base_model, params
