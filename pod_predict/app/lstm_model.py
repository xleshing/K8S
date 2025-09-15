import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pymysql
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim


# ====== DB 連線 ======
def get_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'mysql'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', 'rootpass'),
        database=os.getenv('DB_NAME', 'podmetrics'),
        cursorclass=pymysql.cursors.DictCursor
    )


# ====== LSTM 模型 ======
class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(TimeAwareLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


# ====== 讀取某 pod 歷史資料 ======
def get_pod_data(pod_namespace, pod_name):
    conn = get_connection()
    query = """
        SELECT timestamp, cpu_usage
        FROM pod_cpu_usage
        WHERE pod_namespace = %s AND pod_name = %s
        ORDER BY timestamp;
    """
    df = pd.read_sql(query, conn, params=(pod_namespace, pod_name))
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ====== 時間特徵 ======
def extract_time_features(df):
    df = df.copy()
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    return df


# ====== 建訓練資料（全量） ======
def prepare_data(df, seq_length=30):
    df = extract_time_features(df)
    features = ['cpu_usage', 'month', 'day', 'weekday', 'hour', 'minute']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length][0])  # cpu_usage 維度
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, scaler


# ====== 全量訓練 ======
def train_model(X, y, input_size=6, hidden_size=64, num_layers=2, output_size=1, epochs=100):
    model = TimeAwareLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# ====== 指定時間點預測 ======
def predict_at_time(model, history_df, predict_time_str, scaler, seq_length=30):
    predict_time = pd.to_datetime(predict_time_str)
    last_seq = history_df.tail(seq_length).copy()
    last_seq = extract_time_features(last_seq)
    features = ['cpu_usage', 'month', 'day', 'weekday', 'hour', 'minute']

    # 讓最後一筆的時間特徵等於要預測的時間
    last_seq.iloc[-1, last_seq.columns.get_loc('timestamp')] = predict_time
    last_seq = extract_time_features(last_seq)

    scaled_seq = scaler.transform(last_seq[features])
    input_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(input_seq).item()

    # 反正規化 cpu_usage
    cpu_prediction = scaler.inverse_transform([[prediction, *scaled_seq[-1][1:]]])[0][0]
    return cpu_prediction


# ====== 初始化/存取模型 ======
def init_model_and_optim(input_size=6, hidden_size=64, num_layers=2, lr=1e-3):
    model = TimeAwareLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def save_model(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, path)


def load_model(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])


# ====== Scaler 保存/載入 ======
def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ====== 重新擬合 scaler（滑動視窗） ======
def refit_scaler_with_window(df, feature_cols, window_size=1000):
    window_df = df.tail(window_size).copy()
    scaler = MinMaxScaler()
    scaler.fit(window_df[feature_cols])
    return scaler


# ====== 切序列 ======
def make_sequences_from_df(df, feature_cols, label_col="cpu_usage", seq_len=30, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])

    scaled = scaler.transform(df[feature_cols])
    X_list, y_list = [], []
    total = len(df)
    for i in range(total - seq_len):
        X_list.append(scaled[i:i+seq_len])
        y_list.append(scaled[i+seq_len][0])  # label 用 scaled 的 cpu_usage

    if not X_list:
        return None, None, scaler

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).view(-1, 1)
    return X, y, scaler


# ====== 增量微調 ======
def incremental_finetune(model, optimizer, criterion, df, epochs=1, batch_size=64, seq_len=30, window_size=1000):
    if df is None or df.empty:
        return 0, None

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").copy()
    df = extract_time_features(df)

    feature_cols = ["cpu_usage", "month", "day", "weekday", "hour", "minute"]
    scaler = refit_scaler_with_window(df, feature_cols, window_size=window_size)
    X, y, scaler = make_sequences_from_df(df, feature_cols, label_col="cpu_usage", seq_len=seq_len, scaler=scaler)
    if X is None or X.size(0) == 0:
        return 0, scaler

    model.train()
    n = X.size(0)
    steps = 0
    for _ in range(epochs):
        idx = torch.randperm(n)
        X_shuf, y_shuf = X[idx], y[idx]
        for i in range(0, n, batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            steps += 1
    return steps, scaler