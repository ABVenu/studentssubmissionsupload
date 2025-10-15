from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from config.config import FEATURES

def build_model(units, dropout, lr, seq_len):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True, input_shape=(seq_len, len(FEATURES)))),
        Dropout(dropout),
        GRU(max(1, units // 2), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model
