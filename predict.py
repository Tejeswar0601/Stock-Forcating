"""
Daily Stock Forecast — runs LSTM training + 5-day-ahead prediction for a list
of tickers and writes one JSON file per ticker into predictions/.

This is the notebook's modeling logic, trimmed to run headless (no plots),
looped over multiple tickers, with results saved as JSON for the web page
to read.

Run: python predict.py
"""

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import ta

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ===================== CONFIG =====================
TICKERS = ["AAPL", "TSLA", "MSFT"]   # <-- edit this list to change tracked stocks
START_DATE = "2018-01-01"
SEQ_LEN = 60
FORECAST_HORIZON = 5
TEST_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100

FEATURES = [
    "Daily_Return", "Volume", "EMA_20_Dist", "EMA_50_Dist",
    "MACD", "MACD_Signal", "MACD_Diff", "RSI", "Stoch",
    "BB_PctB", "BB_Width", "ATR_Pct", "OBV_Z", "Vol_Z", "HL_Ratio",
]
TARGET = "Daily_Return"
TARGET_IDX = FEATURES.index(TARGET)
OUTPUT_DIR = "predictions"
# ====================================================


class AttentionLayer(Layer):
    """Bahdanau-style attention over the 60 input timesteps."""

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1), initializer="zeros", trainable=True
        )

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


def add_technical_indicators(df):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    vol = df["Volume"].squeeze()

    ema20 = ta.trend.ema_indicator(close, window=20)
    ema50 = ta.trend.ema_indicator(close, window=50)
    df["EMA_20_Dist"] = (close - ema20) / close
    df["EMA_50_Dist"] = (close - ema50) / close
    df["MACD"] = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]

    df["RSI"] = ta.momentum.rsi(close, window=14)
    df["Stoch"] = ta.momentum.stoch(high, low, close)

    bb = ta.volatility.BollingerBands(close, window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df["BB_PctB"] = (close - bb_lower) / (bb_upper - bb_lower)
    df["BB_Width"] = bb.bollinger_wband()
    df["ATR_Pct"] = ta.volatility.average_true_range(high, low, close) / close

    obv = ta.volume.on_balance_volume(close, vol)
    df["OBV_Z"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
    df["Vol_Z"] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std()

    df["Daily_Return"] = close.pct_change()
    df["HL_Ratio"] = (high - low) / close

    df.dropna(inplace=True)
    return df


def create_sequences(data, seq_len, horizon, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon, target_idx])
    return np.array(X), np.array(y)


def build_model(seq_len, n_features, horizon):
    inp = Input(shape=(seq_len, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = AttentionLayer()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(horizon)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber", metrics=["mae"])
    return model


def predict_future(model, df, scaler, features, seq_len, horizon, target_idx):
    last_seq = df[features].values[-seq_len:]
    last_seq_scaled = scaler.transform(last_seq).reshape(1, seq_len, len(features))
    pred_scaled = model.predict(last_seq_scaled, verbose=0)

    dummy = np.zeros((horizon, len(features)))
    dummy[:, target_idx] = pred_scaled[0]
    future_returns = scaler.inverse_transform(dummy)[:, target_idx]

    last_close = df["Close"].values.flatten()[-1]
    future_prices = np.zeros(horizon)
    cum = last_close
    for d in range(horizon):
        cum = cum * (1 + future_returns[d])
        future_prices[d] = cum
    return future_prices, future_returns


def run_for_ticker(ticker: str) -> dict:
    print(f"\n=== {ticker} ===")
    end_date = datetime.now().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=START_DATE, end=end_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    if len(df) < SEQ_LEN + FORECAST_HORIZON + 100:
        raise ValueError(f"Not enough data for {ticker} ({len(df)} rows)")

    df = add_technical_indicators(df)

    data = df[FEATURES].values
    split = int(len(data) * (1 - TEST_SPLIT))
    train_data = data[:split]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(data[split:])

    X_train, y_train = create_sequences(train_scaled, SEQ_LEN, FORECAST_HORIZON, TARGET_IDX)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN, FORECAST_HORIZON, TARGET_IDX)

    model = build_model(SEQ_LEN, len(FEATURES), FORECAST_HORIZON)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )

    # Hold-out test MAE (on returns) as a rough confidence indicator
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    future_prices, future_returns = predict_future(
        model, df, scaler, FEATURES, SEQ_LEN, FORECAST_HORIZON, TARGET_IDX
    )

    last_price = float(df["Close"].values.flatten()[-1])
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=FORECAST_HORIZON + 1)[1:]

    result = {
        "ticker": ticker,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_known_date": last_date.strftime("%Y-%m-%d"),
        "last_known_price": round(last_price, 2),
        "test_mae_return": round(float(test_mae), 5),
        "forecast": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "day": i + 1,
                "predicted_price": round(float(p), 2),
                "predicted_return_pct": round(float(r) * 100, 3),
                "change_from_last_pct": round(((p - last_price) / last_price) * 100, 2),
            }
            for i, (d, p, r) in enumerate(zip(future_dates, future_prices, future_returns))
        ],
        "recent_history": [
            {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
            for d, c in zip(df.index[-30:], df["Close"].values.flatten()[-30:])
        ],
    }
    print(f"{ticker}: last=${last_price:.2f}  day+1=${future_prices[0]:.2f}  test_mae={test_mae:.5f}")
    return result


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index = []
    for ticker in TICKERS:
        try:
            result = run_for_ticker(ticker)
            out_path = os.path.join(OUTPUT_DIR, f"{ticker}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            index.append({"ticker": ticker, "ok": True})
        except Exception as e:
            print(f"FAILED {ticker}: {e}")
            index.append({"ticker": ticker, "ok": False, "error": str(e)})

    with open(os.path.join(OUTPUT_DIR, "index.json"), "w") as f:
        json.dump(
            {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "tickers": index},
            f,
            indent=2,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
