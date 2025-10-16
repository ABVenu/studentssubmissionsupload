# -*- coding: utf-8 -*-
"""Stock Market Analysis Project


Original file is located at
    https://colab.research.google.com/drive/1J_au8yNwiI7uOnXvFa4PFjOhgxqkntlw
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1. Load the dataset
# ----------------------------
file_path = "/content/Nifty200.csv"
df = pd.read_csv(file_path)

# Convert dates and sort chronologically
df['Date '] = pd.to_datetime(df['Date '], format='%d-%b-%Y')
df = df.sort_values(by='Date ')

# Forward fill missing close prices if any
df['Close '] = df['Close '].fillna(method='ffill')

# ----------------------------
# 2. Prepare the time series
# ----------------------------
series = df['Close '].values.reshape(-1, 1)

# Normalize (mean 0, std 1) to help the model train better
mean, std = series.mean(), series.std()
series = (series - mean) / std

# Create sequences of past N days → predict next day
timesteps = 30
def create_sequences(series, timesteps):
    X, y = [], []
    for i in range(len(series) - timesteps):
        X.append(series[i:i+timesteps])
        y.append(series[i+timesteps])
    return np.array(X), np.array(y)

X, y = create_sequences(series, timesteps)

# Flatten sequences so MLP can use them
X = X.reshape(X.shape[0], -1)

# ----------------------------
# 3. Build the MLP model
# ----------------------------
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ----------------------------
# 4. Train the model
# ----------------------------
history = model.fit(
    X, y,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# 5. Evaluate performance
# ----------------------------
y_pred = model.predict(X)

# Undo normalization
y_pred = y_pred * std + mean
y_true = y * std + mean

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
accuracy = 100 - mape

print("\nModel Performance:")
print(f"MAE   : {mae:.2f} points")
print(f"RMSE  : {rmse:.2f} points")
print(f"R²    : {r2:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"Accuracy ≈ {accuracy:.2f}%")

# ----------------------------
# 6. Plot training history
# ----------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# ----------------------------
# 7. Visualize predictions
# ----------------------------
num_examples = 3
examples_pred = model.predict(X[:num_examples]) * std + mean
examples_true = y[:num_examples] * std + mean
examples_input = X[:num_examples] * std + mean

plt.figure(figsize=(15, 5))
for i in range(num_examples):
    plt.subplot(1, num_examples, i + 1)
    past = examples_input[i].reshape(timesteps)
    plt.plot(range(timesteps), past, marker='o', label="Past Days")
    plt.plot(timesteps, examples_true[i], "bo", label="Actual Next Day")
    plt.plot(timesteps, examples_pred[i], "ro", label="Predicted Next Day")
    plt.title(f"Example {i+1}")
    plt.xlabel("Days")
    plt.ylabel("Nifty Close Price")
    plt.legend()
plt.tight_layout()
plt.show()