import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

app = Flask(__name__)

DATASET_PATH = "."
LSTM_WINDOW = 21
FUTURE_STEPS = 7  # Kaç gün ileri tahmin yapılacak?

COINS = {
    'BTC-USD.csv': 'Bitcoin (BTC)',
    'ETH-USD.csv': 'Ethereum (ETH)',
    'BNB-USD.csv': 'Binance Coin (BNB)',
    'DOT-USD.csv': 'Polkadot (DOT)',
    'DOGE-USD.csv': 'Dogecoin (DOGE)',
    'LTC-USD.csv': 'Litecoin (LTC)',
    'LINK-USD.csv': 'Chainlink (LINK)',
    'XMR-USD.csv': 'Monero (XMR)',
    'ADA-USD.csv': 'Ada (ADA)',
    'XRP-USD.csv': 'XRP (XRP)'
}
MODELS = ['LinearRegression', 'XGBoost', 'LSTM']

STATIC_PLOTS = os.path.join("static", "plots")
os.makedirs(STATIC_PLOTS, exist_ok=True)

def add_features(df):
    df = df.copy()
    df["MA7"] = df["Close"].shift(1).rolling(window=7).mean()
    df["MA21"] = df["Close"].shift(1).rolling(window=21).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).shift(1).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).shift(1).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].shift(1).ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].shift(1).ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    rolling_mean = df["Close"].shift(1).rolling(window=20).mean()
    rolling_std = df["Close"].shift(1).rolling(window=20).std()
    df["BB_up"] = rolling_mean + 2*rolling_std
    df["BB_down"] = rolling_mean - 2*rolling_std
    # Ekstra teknik göstergeler
    low14 = df["Low"].shift(1).rolling(window=14).min()
    high14 = df["High"].shift(1).rolling(window=14).max()
    df["%K"] = 100 * (df["Close"].shift(1) - low14) / (high14 - low14 + 1e-9)
    df["%R"] = -100 * (high14 - df["Close"].shift(1)) / (high14 - low14 + 1e-9)
    df["ATR14"] = (df["High"]-df["Low"]).shift(1).rolling(window=14).mean()
    df["CCI14"] = (df["Close"].shift(1) - rolling_mean) / (0.015 * rolling_std)
    # Lagged features
    for lag in [1,2,3,5,7]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
        df[f"Return_lag{lag}"] = df["Close"].pct_change(lag).shift(1)
    # Zaman kodları
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day
    df = df.bfill()
    return df

def get_feature_list():
    feats = [
        "Open", "High", "Low", "Volume",
        "MA7", "MA21", "RSI14", "MACD", "BB_up", "BB_down",
        "%K", "%R", "ATR14", "CCI14",
        "dayofweek", "month", "day"
    ]
    for lag in [1,2,3,5,7]:
        feats.append(f"Close_lag{lag}")
        feats.append(f"Return_lag{lag}")
    return feats

def split_and_scale(df):
    X = df[get_feature_list()]
    y = df["Close"]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test

def train_linear_regression(df):
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test = split_and_scale(df)
    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_test, y_pred, model, scaler, X_test

def train_xgboost(df):
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test = split_and_scale(df)
    model = XGBRegressor(n_estimators=200, max_depth=5, subsample=0.8, colsample_bytree=0.8, learning_rate=0.08, random_state=0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_test, y_pred, model, scaler, X_test

def train_lstm(df, window=LSTM_WINDOW):
    features = get_feature_list()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df["Close"].values
    def create_sequences(X, y, window=window):
        Xs, ys = [], []
        for i in range(window, len(X)):
            Xs.append(X[i-window:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    split = int(len(df) * 0.8)
    X_seq, y_seq = create_sequences(X_scaled, y)
    X_train, X_test = X_seq[:split-window], X_seq[split-window:]
    y_train, y_test = y_seq[:split-window], y_seq[split-window:]
    all_indices = df.index
    test_indices = all_indices[split:]
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    return y_test, y_pred, model, scaler, test_indices

def direct_multi_step_forecast(df, model_cls, n_steps=FUTURE_STEPS):
    features = get_feature_list()
    preds, trues = [], []
    split = int(len(df) * 0.8)
    scaler = StandardScaler()
    for n in range(1, n_steps+1):
        df_shift = df.copy()
        df_shift["Target"] = df_shift["Close"].shift(-n)
        X_train = df_shift[features].iloc[:split].dropna()
        y_train = df_shift["Target"].iloc[:split].dropna()
        X_test = df_shift[features].iloc[split:-n].dropna()
        y_test = df_shift["Target"].iloc[split:-n].dropna()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = model_cls()
        if hasattr(model, "fit"):
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            pred = model(X_train_scaled, y_train, X_test_scaled)
        preds.append(pred[-1])
        trues.append(y_test.values[-1])
    return preds, trues

def recursive_forecast_next_days(model, df, scaler, steps=FUTURE_STEPS):
    features = get_feature_list()
    df_copy = df.copy()
    future_preds = []
    for i in range(steps):
        last_row = df_copy.iloc[-1].copy()
        new_row = last_row.copy()
        past = df_copy.iloc[-7:] if len(df_copy) >= 7 else df_copy
        open_mean = past["Open"].mean()
        vol_mean = past["Volume"].mean()
        close_std = past["Close"].std() if len(past) > 1 else 0.01*last_row["Close"]
        step_factor = 1 + (i / 3)
        open_noise = np.random.normal(0, 0.002 * open_mean * step_factor)
        new_row["Open"] = 0.7 * last_row["Close"] + 0.3 * open_mean + open_noise
        high_noise = abs(np.random.normal(0, 0.003 * close_std * step_factor))
        low_noise = abs(np.random.normal(0, 0.003 * close_std * step_factor))
        new_row["High"] = new_row["Open"] + high_noise
        new_row["Low"] = max(new_row["Open"] - low_noise, 0.95 * new_row["Open"])
        vol_noise = vol_mean * np.random.uniform(-0.07, 0.07) * step_factor
        new_row["Volume"] = max(vol_mean + vol_noise, 0.1)
        temp_df = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)
        temp_df = add_features(temp_df)
        X_input = temp_df[features].iloc[[-1]]
        X_input_scaled = scaler.transform(X_input)
        pred = model.predict(X_input_scaled)[0]
        future_preds.append(pred)
        temp_df.iloc[-1, temp_df.columns.get_loc("Close")] = pred
        df_copy = temp_df
    return future_preds

def lstm_multi_output_forecast(df, window=LSTM_WINDOW, future_steps=FUTURE_STEPS):
    features = get_feature_list()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df["Close"].values
    Xs, ys = [], []
    for i in range(window, len(X_scaled) - future_steps):
        Xs.append(X_scaled[i-window:i])
        ys.append(y[i:i+future_steps])
    Xs, ys = np.array(Xs), np.array(ys)
    split = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window, len(features))),
        Dropout(0.3),
        RepeatVector(future_steps),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    y_pred = model.predict(X_test)
    future_preds = y_pred[-1].flatten()
    return future_preds

def plot_results(dates, y_test, y_pred, preds_next_days, title, metrics, plot_path_main, plot_path_forecast, step=FUTURE_STEPS):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_test, label='Gerçek')
    plt.plot(dates, y_pred, label='Tahmin')
    plt.title(title + " (Test Seti)")
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    rmse, mae, r2 = metrics
    plt.text(0.98, 0.98,
             f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}',
             ha='right', va='top',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", fc="w", ec="k"))
    plt.tight_layout()
    plt.savefig(plot_path_main)
    plt.close()

    freq = pd.infer_freq(dates)
    if freq is None:
        freq = "D"
    future_dates = pd.date_range(start=dates[-1], periods=step+1, freq=freq)[1:]
    plt.figure(figsize=(8, 4))
    plt.plot([dates[-1]], [y_test[-1]], 'bo', label='Son Gerçek Fiyat')
    plt.plot(future_dates, preds_next_days, 'ro--', label=f"{step} Gün İleri Tahmin")
    plt.title(title + f" ({step} Gün Sonrası Tahmin)")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path_forecast)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    selected_coin = list(COINS.keys())[0]
    selected_model = "LinearRegression"
    plot_file_main = None
    plot_file_forecast = None
    metrics = None

    if request.method == "POST":
        selected_coin = request.form.get("coin")
        selected_model = request.form.get("model")
        print("="*60)
        print(f"Seçili Coin: {COINS[selected_coin]}  |  Model: {selected_model}")

        file_path = os.path.join(DATASET_PATH, selected_coin)
        df = pd.read_csv(file_path)
        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index("Date")
        elif "date" in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index("date")
        elif "Timestamp" in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index("Timestamp")
        df = add_features(df)

        if selected_model == "XGBoost":
            y_test, y_pred, model, scaler, X_test = train_xgboost(df)
            test_index = y_test.index
            preds_next_days, trues_next_days = direct_multi_step_forecast(df, lambda: XGBRegressor(n_estimators=200, max_depth=5, subsample=0.8, colsample_bytree=0.8, learning_rate=0.08, random_state=0), n_steps=FUTURE_STEPS)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = (rmse, mae, r2)
            print(f"XGBoost {FUTURE_STEPS} gün sonrası target shifting tahmin:", [f"{p:.2f}" for p in preds_next_days])
        elif selected_model == "LinearRegression":
            y_test, y_pred, model, scaler, X_test = train_linear_regression(df)
            test_index = y_test.index
            last_pos = df.index.get_loc(test_index[-1])  # Hata-fix!
            df_for_pred = df.iloc[:last_pos+1]
            preds_next_days = recursive_forecast_next_days(model, df_for_pred, scaler, steps=FUTURE_STEPS)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = (rmse, mae, r2)
            print(f"Lineer regresyon recursive {FUTURE_STEPS} günlük tahmin:", [f"{p:.2f}" for p in preds_next_days])
        else:  # LSTM
            y_test, y_pred, model, scaler, test_index = train_lstm(df, window=LSTM_WINDOW)
            preds_next_days = lstm_multi_output_forecast(df, window=LSTM_WINDOW, future_steps=FUTURE_STEPS)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = (rmse, mae, r2)
            print(f"LSTM sequence-to-sequence {FUTURE_STEPS} günlük tahmin:", [f"{p:.2f}" for p in preds_next_days])

        print(f"Test seti boyutu: {len(y_test)}")
        print(f"RMSE: {metrics[0]:.4f} | MAE: {metrics[1]:.4f} | R2: {metrics[2]:.4f}")
        print("="*60)

        plot_file_main = f"{selected_coin.replace('.csv','')}_{selected_model}_main.png"
        plot_file_forecast = f"{selected_coin.replace('.csv','')}_{selected_model}_forecast.png"
        plot_path_main = os.path.join(STATIC_PLOTS, plot_file_main)
        plot_path_forecast = os.path.join(STATIC_PLOTS, plot_file_forecast)
        plot_results(test_index, y_test, y_pred, preds_next_days, f"{COINS[selected_coin]} - {selected_model}", metrics, plot_path_main, plot_path_forecast, step=FUTURE_STEPS)

    return render_template("index.html",
                           coins=COINS,
                           models=MODELS,
                           selected_coin=selected_coin,
                           selected_model=selected_model,
                           plot_file_main=plot_file_main,
                           plot_file_forecast=plot_file_forecast,
                           metrics=metrics,
                           FUTURE_STEPS=FUTURE_STEPS)

@app.route("/static/plots/<filename>")
def plot_file(filename):
    return send_from_directory(STATIC_PLOTS, filename)

if __name__ == "__main__":
    app.run(debug=True)
