import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pymysql
from datetime import datetime

# 定義具有時間特徵的 LSTM 模型
class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(TimeAwareLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# 建立資料庫連線
def get_connection():
    return pymysql.connect(
        host='mysql',
        user='root',
        password='rootpass',
        database='podmetrics',
        cursorclass=pymysql.cursors.DictCursor
    )

# 從資料庫取得指定 Pod 的歷史資料
def get_pod_data(pod_namespace, pod_name):
    conn = get_connection()
    query = """
        SELECT timestamp, cpu_usage FROM pod_cpu_usage
        WHERE pod_namespace = %s AND pod_name = %s
        ORDER BY timestamp;
    """
    df = pd.read_sql(query, conn, params=(pod_namespace, pod_name))
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 擷取時間特徵欄位
def extract_time_features(df):
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    return df

# 建立 LSTM 訓練資料
def prepare_data(df, seq_length=30):
    df = extract_time_features(df)
    features = ['cpu_usage', 'month', 'day', 'weekday', 'hour', 'minute']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length][0])  # 預測 cpu_usage
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, scaler

# 訓練 LSTM 模型
def train_model(X, y, input_size=6, hidden_size=64, num_layers=2, output_size=1, epochs=100):
    model = TimeAwareLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# 預測某一個具體時間點的 CPU 使用率
def predict_at_time(model, history_df, predict_time_str, scaler, seq_length=30):
    predict_time = pd.to_datetime(predict_time_str)
    last_seq = history_df.tail(seq_length).copy()
    last_seq = extract_time_features(last_seq)
    features = ['cpu_usage', 'month', 'day', 'weekday', 'hour', 'minute']

    # 用指定時間替換最後一筆 timestamp，讓模型看到的是目標時刻的時間特徵
    last_seq.iloc[-1, last_seq.columns.get_loc('timestamp')] = predict_time
    last_seq = extract_time_features(last_seq)

    scaled_seq = scaler.transform(last_seq[features])
    input_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(input_seq).item()

    # 還原 cpu_usage 值（反正規化）
    cpu_prediction = scaler.inverse_transform([[prediction, *scaled_seq[-1][1:]]])[0][0]
    return cpu_prediction