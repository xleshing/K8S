# trainer.py
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from lstm_model import (
    get_connection,     # 從 lstm_model 取得 DB 連線
    get_pod_data,       # 讀取單一 pod 的歷史資料 -> DataFrame(timestamp, cpu_usage, ...)
    prepare_data,       # 整理特徵與標籤 -> (X, y, scaler)
    train_model,        # 訓練 LSTM 模型 -> model
    predict_at_time     # 在指定時間點做一次預測 -> float
)

MODEL_DIR = "/models"
INTERVAL = 600  # 每 10 分鐘跑一次
PREDICT_MINUTES = 10  # 預測 10 分鐘後（可自行調整）

def fetch_all_pods():
    """從資料庫撈出目前出現過的所有 (namespace, pod_name)。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT namespace, pod_name FROM pod_cpu_usage")
            rows = cur.fetchall()
        # rows 是 list[dict]，鍵名依你的連線 cursorclass（DictCursor）而定
        pods = [(r["namespace"], r["pod_name"]) for r in rows]
        return pods
    finally:
        conn.close()

def predict_for_pod(namespace: str, pod_name: str):
    """對單一 pod 訓練並預測下一個時間點。回傳 (ok, message)。"""
    df = get_pod_data(namespace, pod_name)
    if df is None or df.empty:
        return False, f"{namespace}/{pod_name}: no data"

    # 取最後一筆時間，預測未來 PREDICT_MINUTES 分鐘
    # 若 df['timestamp'] 不是 datetime 型別，先轉換
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_time_dt = df["timestamp"].max() + pd.Timedelta(minutes=PREDICT_MINUTES)
    target_time_str = target_time_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 準備資料並訓練
    X, y, scaler = prepare_data(df)
    if X is None or len(X) == 0:
        return False, f"{namespace}/{pod_name}: not enough samples"

    model = train_model(X, y)

    # 做一次點預測
    try:
        pred = predict_at_time(model, df, target_time_str, scaler)
    except Exception as e:
        return False, f"{namespace}/{pod_name}: predict error: {e}"

    return True, f"{namespace}/{pod_name} -> predict @ {target_time_str}: cpu={pred:.6f}"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    while True:
        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            pods = fetch_all_pods()
            print(f"[{start_ts}] Found {len(pods)} pods in cluster (from DB).")
            for ns, name in pods:
                ok, msg = predict_for_pod(ns, name)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR (cluster loop): {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()